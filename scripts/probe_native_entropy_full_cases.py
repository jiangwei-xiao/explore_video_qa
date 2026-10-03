"""固定全量得失面板的熵诊断；仅补10次prefill，不生成新答案或执行正式路由。"""
from pathlib import Path
import os
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from probe_native_answer_confidence import score,SOURCE
from videoqa_runtime.common import read_json,sha256,offline_environment
from videoqa_full.state import durable,exclusive,utc
from videoqa_budget32.frames import FrameProvider
RUN=ROOT/'outputs/diagnostics/native_entropy_full_cases6_20261003_r1'
EXP12=ROOT/'outputs/full_videoqa/rd12_exp12__qwen32__videomme2700__20261003_r1'
EXP07=ROOT/'outputs/full_videoqa/rd12_exp07__qwen32__videomme2700__20260930_r1'
PRIOR=ROOT/'outputs/diagnostics/native_confidence_controls2_20261003_r1'


def result_path(root,method,q):
    """接口：区分导入与新结果，要求唯一来源，不在不同批次答案中择优。"""
    candidates=[root/d/method/(q+'.json') for d in ['results','imported/results']]
    found=[p for p in candidates if p.exists()]
    if len(found)!=1:raise ValueError('Ambiguous result identity')
    return found[0]


def main():
    """接口：六题来自熵探索前冻结的面板，629-3复用，只补余下五题两方法。"""
    from videoqa_budget32.run import verify
    with exclusive(RUN/'run.lock'):
        verify(SOURCE)
        manifest=EXP12/'case_manifest.json'
        ids=[r['question_id'] for r in read_json(manifest)['entries'] if r['comparison']=='RD-1.2-exp07']
        assert ids==['007-1','065-3','307-1','301-3','606-3','629-3']
        paths={(q,c):result_path(root,'RD-1.2-'+c,q) for q in ids for c,root in [('exp07',EXP07),('exp12',EXP12)]}
        warm=sorted((SOURCE/'synthetic').glob('*.json'));assert len(warm)==1
        cached={c:PRIOR/'calls'/c/'629-3.json' for c in ['exp07','exp12']}
        # 步骤1：来源和唯一指标先冻结，不新增题型条件、温度校准或得分组合。
        files=[Path(__file__),ROOT/'scripts/probe_native_answer_confidence.py',manifest,warm[0],SOURCE/'protocol.json']+list(paths.values())+list(cached.values())
        spec=dict(question_ids=ids,sources={str(p):sha256(p) for p in files},
            max_real_forward_calls=10,max_synthetic_forward_calls=1,reused_forward_calls=2,
            generation_calls=0,decision='lower natural-log entropy of conditional ABCD distribution; exact ties choose exp12',
            scope='Outcome-selected pre-existing six-case panel; not unbiased accuracy or same-cost method evaluation',
            stop='No additional questions, new confidence metrics, prompt changes or automatic full test')
        pp=RUN/'protocol.json'
        if pp.exists():assert read_json(pp)==spec
        else:durable(pp,spec)
        fp=sha256(pp);schedule=[]
        for j,q in enumerate(ids[:-1]):
            schedule.extend((q,c) for c in (['exp07','exp12'] if j%2==0 else ['exp12','exp07']))
        existing=list((RUN/'calls').glob('*/*.json'))
        assert len(existing)<=10 and all((p.stem,p.parent.name) in schedule for p in existing)
        for p in existing:
            r=read_json(p);assert r['status']=='completed' and r['matches_historical_first_token'] and r['identity']['protocol_sha256']==fp
        for q in ids:
            a,b=[read_json(paths[q,c]) for c in ['exp07','exp12']]
            assert a['question']==b['question'] and a['options']==b['options'] and a['correct']!=b['correct']
        pending=[(q,c) for q,c in schedule if not (RUN/'calls'/c/(q+'.json')).exists()]
        if pending:
            from videoqa_runtime.baseline_run import available_gpus
            free,_=available_gpus(['1']);assert '1' in free
            offline_environment()
            import torch
            from PIL import Image
            from videoqa_full.qwen_fa2_backend import QwenFA2VideoBackend
            torch.set_num_threads(2);torch.set_num_interop_threads(1)
            durable(RUN/'status.json',dict(status='loading',pid=os.getpid(),sid=os.getsid(0),utc=utc()))
            backend=QwenFA2VideoBackend()
            # 步骤2：历史输入身份及首Token均须复现；已启动失败不重试。
            images=[Image.new('RGB',(896,672),(40,i*7,150)) for i in range(32)]
            control=score(backend,images,read_json(warm[0])['result']['observed'],RUN/'synthetic.json',dict(kind='synthetic',protocol_sha256=fp))
            assert control['matches_historical_first_token']
            for im in images:im.close()
            for q,c in pending:
                result=read_json(paths[q,c]);images,hashes=FrameProvider(result['selection']).decode();assert hashes==result['rgb_sha256']
                durable(RUN/'status.json',dict(status='running',question_id=q,condition=c,pid=os.getpid(),sid=os.getsid(0),utc=utc()))
                score(backend,images,result['answer']['observed'],RUN/'calls'/c/(q+'.json'),dict(question_id=q,condition=c,protocol_sha256=fp))
                for im in images:im.close()
        # 步骤3：只做事后方向核验，比较标签不进入熵规则，不产生新问答成绩。
        records=[]
        for q in ids:
            measured={c:read_json(cached[c] if q=='629-3' else RUN/'calls'/c/(q+'.json')) for c in ['exp07','exp12']}
            for c,r in measured.items():
                assert r['matches_historical_first_token'] and r['pixel_sha256']==read_json(paths[q,c])['answer']['observed']['processor_pixel_sha256']
            choice='exp07' if measured['exp07']['entropy']<measured['exp12']['entropy'] else 'exp12'
            records.append(dict(question_id=q,entropy={c:r['entropy'] for c,r in measured.items()},
                choice=choice,chosen_historical_answer_correct=read_json(paths[q,choice])['correct'],
                source='reused_control' if q=='629-3' else 'new_forward'))
        assert len(list((RUN/'calls').glob('*/*.json')))==10
        durable(RUN/'summary.json',dict(records=records,real_forward_calls=10,reused_forward_calls=2,new_generations=0,
            directions_matching_correct_input=sum(r['chosen_historical_answer_correct'] for r in records),
            limitation='Diagnostic panel count, not algorithm accuracy; all six selected for outcome disagreement'))
        durable(RUN/'status.json',dict(status='completed',pid=os.getpid(),sid=os.getsid(0),utc=utc()));print(records,flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        durable(RUN/'status.json',dict(status='paused',error=repr(exc),pid=os.getpid(),utc=utc()));raise
