"""独立后台完成审计：等待原任务结束后核验；不会恢复或追加任何问答。"""
import argparse
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import time
import types

os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
                  HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def scientific_files(root):
    """接口：明确科学产物清单，排除日志/心跳/锁和审计清单自身，避免循环摘要。"""
    root=Path(root);paths=[]
    for folder in ('code_snapshot','offline','selections','traces','posthoc_structure','review'):
        paths.extend(p for p in (root/folder).rglob('*') if p.is_file())
    for profile in ('llava8','qwen32'):
        for folder in ('inputs','calls','raw_generations','results','synthetic','workers'):
            paths.extend(p for p in (root/'qa'/profile/folder).rglob('*') if p.is_file())
        for name in ('protocol.json','prepare_binding.json','processor_boundaries.json','completion_audit.json','execution.json','gpu_allocation.json'):
            p=root/'qa'/profile/name
            if p.exists():paths.append(p)
        p=root/'independent_audit'/(profile+'.json')
        if p.exists():paths.append(p)
    for name in ('offline_protocol.json','offline_summary.json','call_plan.json','summary.json','completion_audit.json',
                 'timing_boundaries.json','preqa_physical_gpu_inventory.json','ssh_detachment_acceptance.json'):
        p=root/name
        if p.exists():paths.append(p)
    return sorted(set(paths))


def inventory(root):
    """接口：记录实际字节摘要、长度及mtime，不重扫模型、视频或特征大文件。"""
    from videoqa_audit import selection_structure_run as study
    root=Path(root);result={}
    for p in scientific_files(root):
        st=p.stat();result[str(p.relative_to(root))]=dict(sha256=study.sha(p),bytes=st.st_size,mtime_ns=st.st_mtime_ns)
    return result


def guarded_restore(root):
    """接口：在真实完成态执行原CLI主函数，模型/选择/解码入口一旦被调用立即失败。"""
    from videoqa_audit import selection_structure as method
    from videoqa_audit import selection_structure_run as study
    from videoqa_audit import selection_structure_runtime as runtime
    from videoqa_generalization import component_ablation_runtime as prior
    from videoqa_full import state
    root=Path(root)
    if not (root/'completion_audit.json').exists():raise ValueError('Completed original run required')
    def forbidden(*args,**kwargs):raise AssertionError('Completed restore attempted generation, selection, decode or model load')
    # 步骤1禁止生产计算路径和模型构造；只允许完成审计的读取与计分重建。
    for module,names in ((method,('select_coarse_control','reference_core')),
            (study,('offline','offline_one')),(runtime,('prepare','execute','worker','process_question')),
            (prior,('cpu_tools',)),(state,('invoke_once',))):
        for name in names:setattr(module,name,forbidden)
    from videoqa_budget32.frames import FrameProvider
    FrameProvider.decode=forbidden
    for name,klass in [('videoqa_runtime.llava_backend','LlavaBackend'),('videoqa_full.qwen_fa2_backend','QwenFA2VideoBackend')]:
        module=types.ModuleType(name);setattr(module,klass,forbidden);sys.modules[name]=module
    path=ROOT/'scripts/run_selection_structure_diagnostic.py'
    spec=importlib.util.spec_from_file_location('structure_guarded_completed_cli',path)
    entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
    sys.argv=[str(path),'--run-id',root.name,'--stage','all','--resume']
    entry.main()


def run(root):
    """接口：后台等待完整终态，再独立统计/解码审计和两次完成恢复；新增模型调用0。"""
    from videoqa_audit import selection_structure_run as study
    from videoqa_full.state import exclusive,durable
    root=Path(root).resolve()
    if not root.is_relative_to((ROOT/'outputs/diagnostics').resolve()):raise ValueError('Unexpected run root')
    audit_dir=root/'independent_audit';audit_dir.mkdir(exist_ok=True)
    files=[Path(__file__),ROOT/'scripts/audit_selection_structure_completion.py',ROOT/'scripts/summarize_selection_structure_traces.py']
    spec=dict(run_dir=str(root),files={str(p.relative_to(ROOT)):study.sha(p) for p in files},new_model_calls=0)
    with exclusive(audit_dir/'coordinator.lock'):
        study.freeze(audit_dir/'coordinator_spec.json',spec)
        if (audit_dir/'coordinator_acceptance.json').exists():
            value=study.read(audit_dir/'coordinator_acceptance.json')
            for relative,old in study.read(audit_dir/'restore_before.json').items():
                p=root/relative;st=p.stat()
                if study.sha(p)!=old['sha256'] or st.st_size!=old['bytes'] or st.st_mtime_ns!=old['mtime_ns']:
                    raise ValueError('Completed audited artifact changed')
            return value
        # 步骤2只观察原PID/终态，暂停、缺失协调器或指纹变化时停止；不重启研究任务。
        while True:
            for p,h in spec['files'].items():
                if study.sha(ROOT/p)!=h:raise ValueError('Independent auditor changed')
            status=study.read(root/'pipeline_status.json')
            if status['status']=='paused':raise ValueError('Original pipeline paused; independent completion audit cannot proceed')
            if status['status']=='completed' and (root/'completion_audit.json').exists():break
            pid=study.read(root/'launcher_process.json')['pid']
            if not Path('/proc',str(pid)).exists():raise ValueError('Original coordinator missing before completion; inspect preserved state')
            durable(audit_dir/'coordinator_status.json',dict(status='waiting',original_pid=pid,
                stage=status.get('stage'),pid=os.getpid(),sid=os.getsid(0),utc=study.utc()))
            time.sleep(15)
        python=study.PYTHONS['qwen32']
        commands=[('full_structure',[python,str(files[2]),'--run-dir',str(root)])]
        commands += [(p+'_raw',[study.PYTHONS[p],str(files[1]),'--run-dir',str(root),'--profile',p]) for p in study.PROFILES]
        commands += [('guarded_restore',[python,str(Path(__file__)),'--run-dir',str(root),'--mode','guarded-restore']),
            ('default_restore',[python,str(ROOT/'scripts/run_selection_structure_diagnostic.py'),'--run-id',root.name,'--stage','all','--resume'])]
        before=None
        for name,command in commands:
            if name=='guarded_restore':
                before=inventory(root);study.freeze(audit_dir/'restore_before.json',before)
            durable(audit_dir/'coordinator_status.json',dict(status='running',stage=name,pid=os.getpid(),sid=os.getsid(0),utc=study.utc()))
            with (audit_dir/(name+'.log')).open('ab',buffering=0) as log:
                subprocess.run(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
            if name in ('guarded_restore','default_restore') and inventory(root)!=before:
                raise ValueError('Completed CLI restore modified scientific artifacts')
        # 步骤3全部原始返回和统计审计通过才发布结果；人工画面复核仍由研究报告单列。
        calls=sum(len(list((root/'qa'/p/'calls').glob('*/*.json'))) for p in study.PROFILES)
        value=dict(status='passed',original_run=str(root),audited_profiles=list(study.PROFILES),
            preserved_scientific_files=len(before),default_cli_restore_passed=True,guarded_cli_restore_passed=True,
            restore_new_generations=0,restore_new_selections=0,restore_model_loads=0,
            original_physical_calls=calls,manual_semantic_review='separate_pending',spec_sha256=study.sha(audit_dir/'coordinator_spec.json'))
        study.freeze(audit_dir/'coordinator_acceptance.json',value)
        durable(audit_dir/'coordinator_status.json',dict(status='completed',pid=os.getpid(),sid=os.getsid(0),utc=study.utc()))
        return value


def main():
    """接口：仅提供独立完成审计或其内部恢复验收模式，不提供生成入口。"""
    import json
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',required=True)
    parser.add_argument('--mode',choices=['run','guarded-restore'],default='run');args=parser.parse_args()
    root=Path(args.run_dir).resolve()
    if not root.is_relative_to((ROOT/'outputs/diagnostics').resolve()):parser.error('Unexpected run root')
    if args.mode=='guarded-restore':guarded_restore(root)
    else:
        try:print(json.dumps(run(root),ensure_ascii=False))
        except BaseException as exc:
            from videoqa_audit import selection_structure_run as study
            from videoqa_full.state import durable
            import traceback
            destination=root/'independent_audit'
            durable(destination/'coordinator_failure.json',dict(status='paused',error=repr(exc),
                traceback=traceback.format_exc(),pid=os.getpid(),sid=os.getsid(0),utc=study.utc(),new_model_calls=0))
            durable(destination/'coordinator_status.json',dict(status='paused',error=repr(exc),utc=study.utc()))
            raise


if __name__=='__main__':main()
