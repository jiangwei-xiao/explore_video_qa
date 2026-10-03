"""原生视频置信度的固定反例控制，四次prefill，不追加生成或选择新帧。"""
from pathlib import Path
import os
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from probe_native_answer_confidence import score,SOURCE
from videoqa_runtime.common import read_json,sha256,offline_environment
from videoqa_full.state import durable,exclusive,utc
from videoqa_budget32.frames import FrameProvider
RUN=ROOT/'outputs/diagnostics/native_confidence_controls2_20261003_r1'
REF=ROOT/'outputs/full_videoqa/rd12_exp07__qwen32__videomme2700__20260930_r1'
EXP12=ROOT/'outputs/full_videoqa/rd12_exp12__qwen32__videomme2700__20261003_r1'
EXP19=ROOT/'outputs/diagnostics/rd12_exp19__qwen32__dev200__20261003_r2'
JOBS=[('383-2','exp07',REF/'imported/results/RD-1.2-exp07/383-2.json'),
      ('383-2','exp19',EXP19/'results/RD-1.2-exp19/383-2.json'),
      ('629-3','exp12',EXP12/'results/RD-1.2-exp12/629-3.json'),
      ('629-3','exp07',REF/'results/RD-1.2-exp07/629-3.json')]


def main():
    """接口：先冻结已知退化对照，再测置信度；结果不可用于挑选更有利的反例题。"""
    from videoqa_budget32.run import verify
    with exclusive(RUN/'run.lock'):
        verify(SOURCE)
        warm=sorted((SOURCE/'synthetic').glob('*.json'));assert len(warm)==1
        paths=[Path(__file__),ROOT/'scripts/probe_native_answer_confidence.py',warm[0],SOURCE/'protocol.json']+[p for _,_,p in JOBS]
        protocol=dict(sources={str(p):sha256(p) for p in paths},jobs=[dict(question_id=q,condition=c,source=str(p)) for q,c,p in JOBS],
            max_real_forward_calls=4,max_synthetic_forward_calls=1,max_generations=0,
            question_selection='Previously reviewed regressions in fixed 12-question panel: 383 direct relation loss and 629 frequency regression; outcome-selected negative controls, not unbiased validation',
            hypothesis='Higher confidence/lower entropy should not systematically favor known wrong input if used as evidence utility',
            stop='No prompt change, additional cases or automatic selector adoption')
        pp=RUN/'protocol.json'
        if pp.exists():assert read_json(pp)==protocol
        else:durable(pp,protocol)
        fp=sha256(pp);allowed={(q,c) for q,c,_ in JOBS};files=list((RUN/'calls').glob('*/*.json'))
        assert len(files)<=4 and all((p.stem,p.parent.name) in allowed for p in files)
        for p in files:
            r=read_json(p);assert r['status']=='completed' and r['matches_historical_first_token'] and r['identity']['protocol_sha256']==fp
        pending=[job for job in JOBS if not (RUN/'calls'/job[1]/(job[0]+'.json')).exists()]
        # 步骤1：同题文本及像素来源固定，标准答案不进入前向输入。
        for q in ['383-2','629-3']:
            values=[read_json(p) for qq,_,p in JOBS if qq==q]
            assert values[0]['question']==values[1]['question'] and values[0]['options']==values[1]['options']
            assert {v['correct'] for v in values}=={True,False}
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
            # 步骤2：复用已验证的原生接口；合成一致后才执行最多四次真实前向。
            images=[Image.new('RGB',(896,672),(40,i*7,150)) for i in range(32)]
            score(backend,images,read_json(warm[0])['result']['observed'],RUN/'synthetic.json',dict(protocol_sha256=fp,kind='synthetic'))
            for im in images:im.close()
            for q,c,p in pending:
                result=read_json(p);images,hashes=FrameProvider(result['selection']).decode();assert hashes==result['rgb_sha256']
                durable(RUN/'status.json',dict(status='running',question_id=q,condition=c,pid=os.getpid(),sid=os.getsid(0),utc=utc()))
                score(backend,images,result['answer']['observed'],RUN/'calls'/c/(q+'.json'),dict(protocol_sha256=fp,question_id=q,condition=c))
                for im in images:im.close()
        # 步骤3：按事前熵和最高概率汇总，不使用正确选项概率作为无监督评分。
        pairs=[]
        for q,changed in [('383-2','exp19'),('629-3','exp12')]:
            a=read_json(RUN/'calls/exp07'/(q+'.json'));b=read_json(RUN/'calls'/changed/(q+'.json'))
            pairs.append(dict(question_id=q,correct_condition='exp07',wrong_condition=changed,
                correct_entropy=a['entropy'],wrong_entropy=b['entropy'],
                correct_confidence=a['top_probability'],wrong_confidence=b['top_probability'],
                wrong_has_lower_entropy=b['entropy']<a['entropy'],wrong_has_higher_confidence=b['top_probability']>a['top_probability']))
        assert len(list((RUN/'calls').glob('*/*.json')))==4
        durable(RUN/'summary.json',dict(pairs=pairs,real_forward_calls=4,new_generations=0,scope='Fixed posthoc negative controls; not an accuracy estimate'))
        durable(RUN/'status.json',dict(status='completed',pid=os.getpid(),sid=os.getsid(0),utc=utc()));print(pairs,flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        durable(RUN/'status.json',dict(status='paused',error=repr(exc),pid=os.getpid(),utc=utc()));raise
