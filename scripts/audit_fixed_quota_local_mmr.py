"""独立完成审计：从真实原始返回和冻结标签重算，不触发任何模型生成。"""
import argparse
import math
import os
from pathlib import Path
import sys

os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
                  HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def independent_pair(data,left,right):
    """接口：独立重建四格及分层视频簇区间；只读取已完成原始返回的计分。"""
    import numpy as np
    rng=np.random.default_rng(2027);numerator=np.zeros(10000);denominator=np.zeros(10000)
    cells={k:[] for k in ('both_correct','left_only','right_only','both_wrong')}
    # 步骤1先按冻结题序归入四格；步骤2独立按时长/视频重抽，不使用报告的对错字段。
    for row in data:
        q=row['question_id'];a,b=left[q],right[q]
        key='both_correct' if a and b else 'left_only' if a else 'right_only' if b else 'both_wrong'
        cells[key].append(q)
    for label in ('short','medium','long'):
        clusters={}
        for row in data:
            if row['stratum']==label:
                q=row['question_id'];clusters.setdefault(row['video_id'],[]).append(int(left[q])-int(right[q]))
        if not clusters:continue
        values=list(clusters.values());sums=np.asarray([sum(x) for x in values]);sizes=np.asarray([len(x) for x in values])
        sampled=rng.integers(len(values),size=(10000,len(values)))
        numerator+=sums[sampled].sum(axis=1);denominator+=sizes[sampled].sum(axis=1)
    return dict(n=len(data),cells=cells,difference_pp=100*(len(cells['left_only'])-len(cells['right_only']))/len(data),
        stratified_video_cluster_bootstrap_95=np.percentile(100*numerator/denominator,[2.5,97.5]).tolist())


def independent_wilson(correct,n):
    """接口：从正确数独立复算Wilson区间，单位为0–1。"""
    z=1.959963984540054;p=correct/n;z2=z*z
    center=(p+z2/(2*n))/(1+z2/n)
    width=z*math.sqrt(p*(1-p)/n+z2/(4*n*n))/(1+z2/n)
    return [max(0.,center-width),min(1.,center+width)]


def audit(root,profile):
    """接口：逐后端读取官方Tokenizer、任务解析与物理返回；不加载视觉或语言模型。"""
    import numpy as np
    from transformers import AutoTokenizer
    from videoqa_audit import fixed_quota_local_mmr as method
    from videoqa_audit import fixed_quota_local_mmr_run as study
    from videoqa_audit import fixed_quota_local_mmr_runtime as runtime
    from videoqa_lmms_bridge.common import load_task,task_doc
    root=Path(root);run=root/'qa'/profile
    if not (run/'completion_audit.json').is_file():raise ValueError('Profile has not completed')
    if not (root/'summary.json').is_file():raise ValueError('Global report must be complete before final independent audit')
    spec=study.read(run/'protocol.json');data=study.development_rows();correct={};sources={};returns=0
    inventory=study.read(ROOT/'configs/local_model_inventory.json')
    path=inventory['models']['llava']['path'] if profile=='llava8' else '/home/models/Qwen/Qwen2.5-VL-7B-Instruct'
    tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
    # 步骤1验收固定四个合成输入的实际模型观测，不能仅凭调用完成状态通过。
    synthetics=list((run/'synthetic').glob('*/*.json'))
    expected=[1,7,8,8] if profile=='llava8' else [1,11,31,32]
    if sorted(study.read(p)['source_frames'] for p in synthetics)!=sorted(expected):
        raise ValueError('Synthetic frame cases differ')
    for p in synthetics:
        value=study.read(p);n=value['source_frames'];o=value['answer']['observed']
        if value['status']!='completed' or o['source_frames']!=n or o['generate_calls']!=1 or o['prefill_calls']!=1:
            raise ValueError('Synthetic actual media/generation differs')
        if not 1<=len(o['generated_token_ids'])<=16:raise ValueError('Synthetic generation cap differs')
        if profile=='llava8':
            if o['visual_tokens']!=n*210 or o['max_new_tokens']!=16:raise ValueError('Synthetic LLaVA token count differs')
        else:
            t,h,w=o['video_grid_thw'][0]
            if (t!=(n+1)//2 or o['internal_padding_frames']!=n%2 or o['encoding_fps']!=2
                    or o['visual_tokens']!=t*h*w//4 or o['generation_kwargs']['max_new_tokens']!=16):
                raise ValueError('Synthetic Qwen temporal patch/token count differs')
            if n in (31,32) and h*w*196!=602112:raise ValueError('Synthetic Qwen did not test pixel upper boundary')
    for m in (method.PARENT,method.CONTROL,method.GLOBAL,method.WFS):
        correct[m]={}
        for row in data:
            q=row['question_id'];source_run=run if m in (method.PARENT,method.CONTROL) else study.PREVIOUS/'qa'/profile
            result_path=source_run/'results'/m/(q+'.json');value=study.read(result_path)
            item=study.read(source_run/'inputs'/(q+'.json'))['items'][m]
            if value['selection']!=item['selection'] or value['rgb_sha256']!=item['rgb_sha256']:
                raise ValueError('Result/input source identity differs')
            bundle=runtime.physical_sources(result_path,value);raws={}
            for p,h in bundle.items():
                if '/raw_generations/' in p:raws.setdefault(h,study.read(p))
            matching=[v for v in raws.values() if v['raw_generated_text'].strip()==value['answer']['raw_output']
                      and v['observed']['generated_token_ids']==value['answer']['observed']['generated_token_ids']]
            if not matching:raise ValueError('No matching confirmed original return')
            for raw in matching:
                if raw['generation_state']!=dict(started=True,returned=True):raise ValueError('Uncertain raw return')
                tokens=raw['observed']['generated_token_ids']
                if not 1<=len(tokens)<=16 or tokenizer.decode(tokens,skip_special_tokens=True).strip()!=raw['raw_generated_text'].strip():
                    raise ValueError('Original token/text decode differs')
            metric=load_task().videomme_process_results(task_doc(row),[matching[0]['raw_generated_text'].strip()])['videomme_perception_score']
            hit=metric['pred_answer']=='ABCD'[row['answer_index']]
            if hit!=value['correct'] or metric!=value['answer']['framework_metric']:raise ValueError('Independent raw correctness differs')
            correct[m][q]=hit;returns+=1;sources[str(result_path)]=study.sha(result_path);sources.update(bundle)
    value=dict(status='passed',profile=profile,method_results=returns,correct={m:sum(v.values()) for m,v in correct.items()},
        source_hashes=sources,auditor_sha256=study.sha(Path(__file__)),new_model_calls=0)
    # 步骤3全局报告已生成时，再独立核对全部六项配对及总体Wilson；不修改原报告。
    if (root/'summary.json').exists():
        summary=study.read(root/'summary.json');pairs={}
        for other in (method.PARENT,method.GLOBAL,method.WFS):
            for name,subset in [('all200',data),('excluding_previous_panel192',[r for r in data if r['question_id'] not in study.OLD_PANEL])]:
                pair=independent_pair(subset,correct[method.CONTROL],correct[other])
                saved=summary['profiles'][profile]['comparisons'][other][name]
                if pair!=saved: raise ValueError('Independent paired bootstrap differs')
                pairs[other+'/'+name]=pair
        for m,bank in correct.items():
            saved=summary['profiles'][profile]['metrics'][m]['all200']
            if saved['correct']!=sum(bank.values()) or not np.allclose(independent_wilson(sum(bank.values()),len(data)),saved['wilson_95'],rtol=0,atol=1e-12):
                raise ValueError('Independent Wilson/count differs')
        value['comparisons']=pairs;value['summary_sha256']=study.sha(root/'summary.json')
    study.freeze(root/'independent_audit'/(profile+'.json'),value)
    return {k:v for k,v in value.items() if k not in {'source_hashes','comparisons'}}


def guarded_restore(root):
    """接口：在真实完成CLI中封锁模型、解码、选择与生成，确保完成恢复纯核验。"""
    import importlib.util
    import types
    from videoqa_audit import fixed_quota_local_mmr as method
    from videoqa_audit import fixed_quota_local_mmr_run as study
    from videoqa_audit import fixed_quota_local_mmr_runtime as runtime
    from videoqa_generalization import component_ablation_runtime as prior
    from videoqa_full import state
    from videoqa_budget32.frames import FrameProvider
    def forbidden(*args,**kwargs):
        raise AssertionError('Completed restore attempted model load, selection, decode or generation')
    # 步骤1封锁实际计算边界；步骤2调用真实CLI而不是伪造审计返回。
    for module,names in [(method,('select_region_mmr_fixed_quota','reference_core')),
            (study,('offline','offline_one')),(runtime,('prepare','execute','worker','process_question')),
            (prior,('cpu_tools',)),(state,('invoke_once',))]:
        for name in names:setattr(module,name,forbidden)
    FrameProvider.decode=forbidden
    for name,klass in [('videoqa_runtime.llava_backend','LlavaBackend'),('videoqa_full.qwen_fa2_backend','QwenFA2VideoBackend')]:
        fake=types.ModuleType(name);setattr(fake,klass,forbidden);sys.modules[name]=fake
    path=ROOT/'scripts/run_fixed_quota_local_mmr.py'
    spec=importlib.util.spec_from_file_location('fixed_mmr_completed_cli',path)
    entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
    sys.argv=[str(path),'--run-id',root.name,'--stage','all','--resume'];entry.main()


def inventory(root):
    """接口：核验科学产物字节/大小/mtime，日志心跳和审计清单不进入循环摘要。"""
    import run_selection_structure_independent_audit as previous
    result=previous.inventory(root)
    for name in ('ssh_disconnect_acceptance.json',):
        p=root/name
        if p.exists():
            from videoqa_audit.fixed_quota_local_mmr_run import sha
            st=p.stat();result[name]=dict(sha256=sha(p),bytes=st.st_size,mtime_ns=st.st_mtime_ns)
    return result


def run(root):
    """接口：独立后台等待正式终态，复算原文与区间并验证两种零新增恢复。"""
    import subprocess
    import time
    from videoqa_audit import fixed_quota_local_mmr_run as study
    from videoqa_full.state import exclusive,durable
    root=Path(root);directory=root/'independent_audit'
    with exclusive(directory/'coordinator.lock'):
        spec=dict(auditor_sha256=study.sha(Path(__file__)),new_model_calls=0)
        study.freeze(directory/'spec.json',spec)
        # 步骤1仅观察原协调器，不恢复、重启或追加研究问答。
        while True:
            if study.sha(Path(__file__))!=spec['auditor_sha256']:raise ValueError('Auditor changed')
            status=study.read(root/'pipeline_status.json')
            if status['status']=='paused':raise ValueError('Original pipeline paused')
            if status['status']=='completed' and (root/'completion_audit.json').exists():break
            pid=study.read(root/'launcher.json')['pid']
            if not Path('/proc',str(pid)).exists():raise ValueError('Original coordinator missing')
            durable(directory/'status.json',dict(status='waiting',stage=status.get('stage'),pid=os.getpid(),sid=os.getsid(0),utc=study.utc()))
            time.sleep(15)
        for profile in study.PROFILES:
            with (directory/(profile+'.log')).open('ab',buffering=0) as log:
                subprocess.run([study.PYTHONS[profile],str(Path(__file__)),'--run-dir',str(root),
                    '--mode','raw','--profile',profile],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
        # 步骤2两种真实完成恢复前后快照完全相同；不改写旧结果或科学产物。
        before=inventory(root);study.freeze(directory/'restore_before.json',before)
        commands=[['--mode','guarded'],None]
        for ordinal,tail in enumerate(commands):
            command=([study.PYTHONS['qwen32'],str(Path(__file__)),'--run-dir',str(root),*tail] if tail else
                [study.PYTHONS['qwen32'],str(ROOT/'scripts/run_fixed_quota_local_mmr.py'),'--run-id',root.name,'--stage','all','--resume'])
            with (directory/('restore_'+str(ordinal)+'.log')).open('ab',buffering=0) as log:
                subprocess.run(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
            if inventory(root)!=before:raise ValueError('Completed CLI modified scientific artifacts')
        value=dict(status='passed',preserved_scientific_files=len(before),guarded_restore_passed=True,
            default_restore_passed=True,restore_new_generations=0,restore_model_loads=0,restore_new_selections=0,
            new_model_calls=0,auditor_sha256=spec['auditor_sha256'])
        study.freeze(directory/'acceptance.json',value)
        durable(directory/'status.json',dict(status='completed',pid=os.getpid(),sid=os.getsid(0),utc=study.utc()))
        return value


def main():
    """接口：独立完成审计、单后端原文审计或内部恢复验收；均无生成入口。"""
    import json
    from videoqa_audit import fixed_quota_local_mmr_run as study
    from videoqa_full.state import durable
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',required=True)
    parser.add_argument('--mode',choices=['run','raw','guarded'],default='run')
    parser.add_argument('--profile',choices=study.PROFILES);args=parser.parse_args()
    root=Path(args.run_dir).resolve()
    if not root.is_relative_to((ROOT/'outputs/diagnostics').resolve()):parser.error('Unexpected run root')
    try:
        if args.mode=='raw':value=audit(root,args.profile)
        elif args.mode=='guarded':guarded_restore(root);value=dict(status='passed')
        else:value=run(root)
        print(json.dumps(value,ensure_ascii=False))
    except BaseException as exc:
        import traceback
        durable(root/'independent_audit/failure.json',dict(status='paused',error=repr(exc),traceback=traceback.format_exc(),new_model_calls=0))
        raise


if __name__=='__main__':main()

