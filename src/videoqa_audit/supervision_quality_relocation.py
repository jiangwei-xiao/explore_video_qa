"""用户授权后的未开始题目换卡：保留旧55题，GPU0—3仅执行剩余65题。"""
from collections import Counter
import copy
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

from videoqa_runtime.common import ROOT, read_json, sha256
from videoqa_full.state import durable, exclusive, utc
from videoqa_audit import supervision_quality as d

PARENT=ROOT/'outputs/diagnostics/supervision_quality__nextqa120__llava8__20261019_r1'
GPUS=('0','1','2','3')


def assign_remaining(cases):
    """接口：只按剩余公开题目的时长平衡GPU0—3；不读取历史正确性。"""
    loads={g:0. for g in GPUS};mapping={}
    for case in sorted(cases,key=lambda c:(-c['duration_seconds'],c['ordinal'])):
        gpu=min(GPUS,key=lambda g:(loads[g],g));mapping[case['row']['question_id']]=gpu
        loads[gpu]+=case['duration_seconds']
    return mapping


def verify_parent_handoff(parent=PARENT):
    """接口：确认原协调器已停止、旧调用已返回，并逐字节保护已完成产物。"""
    parent=Path(parent);receipt=read_json(parent/'relocation_handoff.json')
    command=Path('/proc')/str(receipt['stopped_coordinator_pid'])/'cmdline'
    if command.exists() and command.read_bytes():raise RuntimeError('Parent coordinator still live')
    for rel,expected in receipt['artifacts'].items():
        if sha256(parent/rel)!=expected:raise ValueError('Parent handoff artifact changed: '+rel)
    return receipt


def prepare(root):
    """接口：仅迁移未开始题目的执行分配；120题和全部五条件输入保持原字节。"""
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    if (root/'relocation.json').exists():verify(root);return read_json(root/'relocation.json')
    with exclusive(root/'prepare.lock'):
        receipt=verify_parent_handoff();d.verify(PARENT,True)
        panel=read_json(PARENT/'panel.json');evaluation=read_json(PARENT/'evaluation.json');ready=read_json(PARENT/'inputs_complete.json')
        completed={p.stem for p in (PARENT/'results/O').glob('*.json')}
        pending=[c for c in panel['cases'] if c['row']['question_id'] not in completed]
        if len(completed)!=55 or len(pending)!=65:raise ValueError('Handoff population changed')
        for c in pending:
            q=c['row']['question_id']
            if any((PARENT/folder/condition/(q+'.json')).exists() for folder in ('results','qa/calls','qa/raw_generations') for condition in d.CONDITIONS):
                raise ValueError('Only never-started questions may move')
        assignment=assign_remaining(pending);shard=root/'relocated';shard.mkdir(parents=True,exist_ok=True)
        full_evaluation=copy.deepcopy(evaluation);assignment_record={}
        # 步骤1：原输入不重建；原题号/源帧/RGB/提示的冻结JSON逐字节复制。
        for case in panel['cases']:
            q=case['row']['question_id'];target=root/'inputs'/(q+'.json');target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(PARENT/'inputs'/(q+'.json'),target)
            actual=assignment.get(q,evaluation[q]['gpu']);historical=evaluation[q]['gpu']
            full_evaluation[q].update(gpu=actual,historical_gpu=historical)
            assignment_record[q]=dict(historical_gpu=historical,execution_gpu=actual,
                batch='relocated' if q in assignment else 'parent',source_run=str(shard if q in assignment else PARENT))
        shutil.copyfile(PARENT/'panel.json',root/'panel.json');durable(root/'evaluation.json',full_evaluation)
        durable(root/'execution_assignment.json',assignment_record)
        # 步骤2：旧55题只复制方法级结果，call_path仍指向原返回；不复制成新物理调用。
        imported={}
        for q in sorted(completed):
            for condition in d.CONDITIONS:
                old=PARENT/'results'/condition/(q+'.json');dest=root/'results'/condition/(q+'.json')
                dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(old,dest);imported[str(old)]=sha256(old)
        shard_eval={q:full_evaluation[q] for q in assignment}
        durable(shard/'panel.json',dict(cases=pending,question_count=65,video_count=65,sample_uses_outcomes=False,
                    scope='原120题中的未开始题目；没有重新抽样',parent_panel_sha256=sha256(PARENT/'panel.json')))
        durable(shard/'evaluation.json',shard_eval);shutil.copyfile(PARENT/'gray.png',shard/'gray.png')
        exports={}
        for q in assignment:
            dest=shard/'inputs'/(q+'.json');dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(PARENT/'inputs'/(q+'.json'),dest);exports[q]=sha256(dest)
        ceiling=sum(len(read_json(shard/'inputs'/(q+'.json'))['groups']) for q in assignment)
        if ceiling+receipt['original_real_calls']!=599:raise ValueError('Combined real-call ceiling changed')
        original=read_json(PARENT/'protocol.json');code=dict(original['code'])
        for name in ('src/videoqa_audit/supervision_quality_relocation.py','scripts/run_relocated_supervision_quality.py'):
            code[name]=sha256(ROOT/name)
        for name in code:
            dest=root/'code_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
        sources=dict(original['sources']);sources.update(imported)
        for path in (PARENT/'relocation_handoff.json',PARENT/'protocol.json',PARENT/'inputs_complete.json',PARENT/'panel.json',PARENT/'evaluation.json'):
            sources[str(path)]=sha256(path)
        protocol=dict(original,task='supervision_quality_v1_authorized_relocation',code=code,sources=sources,
                      parent_run=str(PARENT),relocation_authorization='用户明确回复：在GPU0~3进行',
                      original_calls_retained=receipt['original_real_calls'],remaining_call_ceiling=ceiling,
                      historical_gpu_requirement_amended=True,utc=utc())
        durable(shard/'protocol.json',dict(protocol,maximum_real_calls=ceiling,maximum_synthetic_calls=4))
        durable(root/'protocol.json',protocol)
        uuid={g:subprocess.check_output(['nvidia-smi','-i',g,'--query-gpu=uuid','--format=csv,noheader'],text=True).strip() for g in GPUS}
        shard_ready=dict(ready,questions=65,condition_results=325,exports=exports,unique_input_call_ceiling=ceiling,
            maximum_real_calls=ceiling,maximum_synthetic_calls=4,panel_sha256=sha256(shard/'panel.json'),
            evaluation_sha256=sha256(shard/'evaluation.json'),gpu_uuids=uuid,
            historical_states=dict(Counter(e['historical_state'] for e in shard_eval.values())),
            preparation_seconds=0.,preparation_provenance='复用r1已经冻结的实际输入，不重复解码或预处理')
        durable(shard/'inputs_complete.json',shard_ready)
        durable(shard/'qa/protocol.json',dict(parent_protocol_sha256=sha256(shard/'protocol.json'),
                inputs_sha256=sha256(shard/'inputs_complete.json'),maximum_real_calls=ceiling))
        durable(shard/'synthetic/protocol.json',dict(parent_protocol_sha256=sha256(shard/'protocol.json'),maximum_calls=4))
        durable(root/'inputs_complete.json',dict(ready,panel_sha256=sha256(root/'panel.json'),
                evaluation_sha256=sha256(root/'evaluation.json'),execution_assignment_sha256=sha256(root/'execution_assignment.json'),gpu_uuids=uuid))
        value=dict(parent_run=str(PARENT),completed_questions=55,remaining_questions=65,imported_condition_results=275,
            original_real_calls=receipt['original_real_calls'],remaining_call_ceiling=ceiling,combined_call_ceiling=599,
            original_synthetic_calls=4,remaining_synthetic_ceiling=4,assignment=assignment,
            unchanged_input_exports=ready['exports'],historical_gpu_requirement_amended=True,utc=utc())
        durable(root/'relocation.json',value);verify(root);return value


def verify(root):
    """接口：恢复与最终审计同时核验旧返回、新执行图和所有实际输入，拒绝扩大调用范围。"""
    root=Path(root);verify_parent_handoff();d.verify(root/'relocated',True)
    ready=read_json(root/'inputs_complete.json');migration=read_json(root/'relocation.json')
    if sha256(root/'panel.json')!=sha256(PARENT/'panel.json'):raise ValueError('Original 120-question panel changed')
    if sha256(root/'evaluation.json')!=ready['evaluation_sha256'] or sha256(root/'execution_assignment.json')!=ready['execution_assignment_sha256']:
        raise ValueError('Execution assignment changed')
    for q,h in migration['unchanged_input_exports'].items():
        if sha256(root/'inputs'/(q+'.json'))!=h:raise ValueError('Original diagnostic input changed')
    if (root/'completion.json').exists():
        for rel,h in read_json(root/'completion.json')['artifacts'].items():
            if sha256(root/rel)!=h:raise ValueError('Completed relocated artifact changed')


def audit(root):
    """接口：跨两个实际批次复算600条结果；保留原call_path、原始返回及真实物理GPU。"""
    from videoqa_set_value.task import strict_option
    from videoqa_audit.set_value_trainability import sealed_artifacts
    root=Path(root);verify(root);mapping=read_json(root/'execution_assignment.json');evaluation=read_json(root/'evaluation.json')
    calls=set();shared=0;skipped=0;reproduced=0;counts={c:0 for c in d.CONDITIONS}
    for case in read_json(root/'panel.json')['cases']:
        q=case['row']['question_id'];assignment=mapping[q];source=Path(assignment['source_run'])
        plan=read_json(root/'inputs'/(q+'.json'));ev=evaluation[q]
        original=read_json(root/'results/O'/(q+'.json'));same=d.historical_reproduced(original['answer']['raw_output'],ev['historical'][d.METHODS[0]]['parsed_answer'])
        reproduced+=same
        for condition in d.CONDITIONS:
            result=read_json(root/'results'/condition/(q+'.json'))
            if result!=read_json(source/'results'/condition/(q+'.json')):raise ValueError('Combined result differs from original source')
            if result['status'].startswith('skipped'):
                if same or condition=='O':raise ValueError('Invalid skip')
                skipped+=1;continue
            if not same and condition!='O':raise ValueError('Condition after nonreproduced O')
            rep=result['representative'];path=source/'qa/calls'/rep/(q+'.json');call=read_json(path)
            raw=read_json(source/'qa/raw_generations'/rep/(q+'.json'));answer=result['answer']
            if Path(result['call_path'])!=path or call['gpu']!=assignment['execution_gpu'] or result['gpu']!=call['gpu']:
                raise ValueError('Actual GPU/call source differs')
            if call['status']!='completed' or call['answer']!=answer or raw['generation_state']!={'started':True,'returned':True}:
                raise ValueError('Incomplete source return')
            if call['protocol_sha256']!=sha256(source/'qa/protocol.json') or raw['identity']!=dict(question_id=q,method=rep,gpu=call['gpu'],protocol_sha256=call['protocol_sha256']):
                raise ValueError('Raw call fingerprint differs')
            d._check_answer(answer,plan['entries'][condition]['identity'])
            if raw['raw_generated_text'].strip()!=answer['raw_output']:raise ValueError('Raw text differs')
            p=strict_option(answer['raw_output'])
            if p!=result['parsed_answer'] or (p=='ABCDE'[ev['answer_index']])!=result['correct']:
                raise ValueError('Independent score differs')
            if result['input_key']!=plan['entries'][condition]['input_key'] or result['input_key']!=plan['entries'][rep]['input_key']:
                raise ValueError('Condition sharing mismatch')
            counts[condition]+=1;shared+=result['shared'];calls.add(str(path))
    physical=[p for base in (PARENT,root/'relocated') for p in (base/'qa/calls').glob('*/*.json')]
    synthetic=[p for base in (PARENT,root/'relocated') for p in (base/'synthetic/calls').glob('*/*.json')]
    if set(map(str,physical))!=calls or len(physical)>599 or len(physical)+shared+skipped!=600 or len(synthetic)>8:
        raise ValueError('Cross-batch call budget does not close')
    if len({read_json(p)['call_id'] for p in physical})!=len(physical):raise ValueError('Repeated cross-batch call ID')
    if any(read_json(p)['status']!='completed' for p in synthetic):raise ValueError('Synthetic failure')
    protection=sealed_artifacts(d.SOURCE)
    if protection!=read_json(root/'protocol.json')['original_protection']:raise ValueError('Original experiment changed')
    return dict(status='passed',questions=120,condition_results=600,real_calls=len(physical),shared_results=shared,
        skipped_results=skipped,synthetic_calls=len(synthetic),reproduced_originals=reproduced,complete_condition_counts=counts,
        imported_questions=55,relocated_questions=65,original_protection=protection,new_training_updates=0,
        new_blip_calls=0,new_refinement_frames=0,new_feature_calls=0,original_validation_evaluated=False)


def run(root):
    """接口：GPU0—3后台完成未开始题目，随后合并、原计分与只读恢复审计。"""
    from videoqa_runtime.baseline_run import available_gpus
    from videoqa_audit.supervision_quality_report import summarize
    root=Path(root);children={}
    with exclusive(root/'coordinator.lock'):
        if (root/'completion.json').exists():return audit(root)
        try:
            if (root/'PAUSE.json').exists():raise RuntimeError('Paused relocated run requires inspection')
            began=time.monotonic();prepare(root);launched=set();completed=set();shard=root/'relocated'
            while len(completed)<4:
                try:free,_=available_gpus(list(GPUS))
                except RuntimeError as error:
                    if str(error)!='No requested GPU is currently idle':raise
                    free=[]
                for gpu in GPUS:
                    if gpu in launched or gpu not in free:continue
                    if (shard/'workers'/(gpu+'.json')).exists():raise RuntimeError('Prior relocated worker session; no automatic replay')
                    log=(root/('worker_'+gpu+'.log')).open('a',buffering=1)
                    command=[d.QA_PYTHON,str(ROOT/'scripts/run_relocated_supervision_quality.py'),'--run-id',root.name,'--stage','worker','--gpu',gpu]
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
                    p=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                    children[gpu]=(p,log);launched.add(gpu)
                for gpu,(p,_) in children.items():
                    code=p.poll()
                    if code not in (None,0):raise RuntimeError('Relocated worker failed: '+gpu)
                    if code==0:
                        if read_json(shard/'workers'/(gpu+'.json'))['status']!='completed':raise RuntimeError('Worker ended before completion')
                        completed.add(gpu)
                if (shard/'PAUSE.json').exists():raise RuntimeError('Relocated worker paused')
                n=sum(len(list((shard/'results'/c).glob('*.json'))) for c in d.CONDITIONS)
                durable(root/'heartbeat.json',dict(stage='teacher',pid=os.getpid(),sid=os.getsid(0),
                    completed_results=275+n,total_results=600,imported_results=275,new_results=n,
                    workers={g:p.pid for g,(p,_) in children.items()},elapsed_seconds=time.monotonic()-began,utc=utc()))
                if len(completed)<4:time.sleep(3)
            # 步骤1：新65题所有结果返回后统一复制展示，保留其实际物理调用路径。
            for path in (shard/'results').glob('*/*.json'):
                dest=root/'results'/path.parent.name/path.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
            checked=audit(root);result=summarize(root,checked)
            result['execution_batches']=read_json(root/'relocation.json');durable(root/'report.json',result)
            durable(root/'audit.json',checked)
            artifacts={str(p.relative_to(root)):sha256(p) for folder in ('inputs','results','relocated/qa','relocated/results','relocated/synthetic')
                       for p in (root/folder).rglob('*') if p.is_file()}
            for name in ('protocol.json','panel.json','evaluation.json','execution_assignment.json','relocation.json','inputs_complete.json','report.json','case_panel.json','index.html','audit.json'):
                artifacts[name]=sha256(root/name)
            durable(root/'completion.json',dict(status='completed',artifacts=artifacts,wall_seconds=time.monotonic()-began,utc=utc()))
            durable(root/'heartbeat.json',dict(stage='completed',pid=os.getpid(),completed_results=600,utc=utc()));return result
        except BaseException as error:
            durable(root/'PAUSE.json',dict(error=repr(error),traceback=traceback.format_exc(),utc=utc()));raise
        finally:
            for p,log in children.values():
                if p.poll() is None:p.wait()
                log.close()
