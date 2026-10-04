"""exp12两题四次上下文修复诊断，复用正式QA，不改变任何方法成绩。"""
import os
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from prepare_context_repairs import RUN
from videoqa_audit.query_input_probe import BASE
from videoqa_budget32.run import verify,process,validate,recover_call
from videoqa_full.state import durable,exclusive,rows,utc
from videoqa_runtime.common import read_json,sha256,offline_environment


def execute():
    """接口：四次真实诊断、一次合成；原复测不一致只暂停归因，不自动重答。"""
    with exclusive(RUN/'coordinator.lock'):
        inputs=read_json(RUN/'input_manifest.json');review=read_json(RUN/'visibility_review.json')
        if inputs['question_ids']!=['541-1','705-1'] or inputs['max_real_generations']!=4:
            raise ValueError('Fixed intervention population changed')
        if review['status']!='passed' or review['input_manifest_sha256']!=sha256(RUN/'input_manifest.json'):
            raise ValueError('Actual-input visibility review missing')
        for path,digest in inputs['sources'].items():
            if sha256(path)!=digest:raise ValueError('Input source changed')
        for rel,digest in inputs['files'].items():
            if sha256(RUN/rel)!=digest:raise ValueError('Frozen input changed')
        # 步骤1：继承同一问答协议，不在提示中加入人工取证说明或答案。
        if not (RUN/'protocol.json').exists():
            spec=dict(verify(BASE),run_id=RUN.name,methods=['original','repair'],
                question_ids=inputs['question_ids'],max_calls=4,max_calls_per_method=2,allowed_gpus=['1'],diagnostic=True,
                selection_exports={k:v for k,v in inputs['files'].items() if k.startswith('selections/')},
                diagnostic_input_sha256=sha256(RUN/'input_manifest.json'),
                visibility_review_sha256=sha256(RUN/'visibility_review.json'),runner_sha256=sha256(__file__),
                scope='Human-directed two-frame context replacement in exp12; not algorithm accuracy')
            durable(RUN/'protocol.json',spec)
        spec=verify(RUN)
        if spec['runner_sha256']!=sha256(__file__) or spec['diagnostic_input_sha256']!=sha256(RUN/'input_manifest.json'):
            raise ValueError('Execution protocol changed')
        data={r['question_id']:r for r in rows()}
        dev=read_json(ROOT/'data/manifests/video_mme_development_200_20260920.json')['rows']
        ordinal={r['question_id']:i for i,r in enumerate(dev)}
        schedule=[(q,c) for q in inputs['question_ids'] for c in
                  (['original','repair'] if ordinal[q]%2==0 else ['repair','original'])]
        calls=list((RUN/'calls').glob('*/*.json'))
        if len(calls)>4 or any((p.stem,p.parent.name) not in schedule for p in calls):raise ValueError('Call budget mismatch')
        for p in calls:recover_call(RUN,data[p.stem],p.parent.name,spec)
        pending=[(q,c) for q,c in schedule if not (RUN/'results'/c/(q+'.json')).exists()]
        # 步骤2：单GPU独立会话，同题两条件同进程；完成恢复不加载模型。
        if pending:
            from videoqa_runtime.baseline_run import available_gpus
            free,_=available_gpus(['1'])
            if '1' not in free:raise RuntimeError('Original physical GPU1 unavailable')
            os.environ['CUDA_VISIBLE_DEVICES']='1';offline_environment()
            import torch
            from PIL import Image
            from videoqa_full.qwen_fa2_backend import QwenFA2VideoBackend
            from videoqa_budget32.qwen_adapter import FrozenInputQwen
            from videoqa_budget32.observe import observe
            from videoqa_lmms_bridge.bridge import synthetic_row
            torch.set_num_threads(2);torch.set_num_interop_threads(1)
            durable(RUN/'status.json',dict(status='loading',pid=os.getpid(),sid=os.getsid(0),utc=utc()))
            backend=QwenFA2VideoBackend();adapter=FrozenInputQwen(backend)
            warm=RUN/'synthetic.json'
            if warm.exists():
                if read_json(warm)['status']!='completed':raise ValueError('Unknown synthetic generation')
            else:
                durable(warm,dict(status='started',utc=utc()))
                images=[Image.new('RGB',(896,672),(40,i*7,150)) for i in range(32)]
                result=observe(backend,adapter,images,synthetic_row(32),{})
                for image in images:image.close()
                durable(warm,dict(status='completed',result=result,synthetic_generations=1))
            for q,c in pending:
                durable(RUN/'status.json',dict(status='running',question_id=q,condition=c,pid=os.getpid(),sid=os.getsid(0),utc=utc()))
                process(RUN,data[q],c,backend,adapter,'1',spec)
                actual=validate(RUN,data[q],c,spec)
                if actual['answer']['observed']['processor_pixel_sha256']!=inputs['observations'][q]['conditions'][c]['processor_pixel_sha256']:
                    raise ValueError('Actual model-visible input differs from frozen intervention')
        # 步骤3：先核对历史复测，再解释组合干预；不因答错或不一致追加生成。
        pairs=[]
        for q in inputs['question_ids']:
            a=validate(RUN,data[q],'original',spec);b=validate(RUN,data[q],'repair',spec)
            old=read_json(inputs['observations'][q]['historical_result'])
            same=a['answer']['parsed_answer']==old['answer']['parsed_answer']
            pairs.append(dict(question_id=q,original=a['answer']['parsed_answer'],repair=b['answer']['parsed_answer'],
                original_correct=a['correct'],repair_correct=b['correct'],historical_replay_matches=same,
                attribution='eligible' if same else 'suspended_replay_mismatch'))
        if len(list((RUN/'calls').glob('*/*.json')))!=4:raise ValueError('Incomplete call ledger')
        durable(RUN/'summary.json',dict(status='completed',actual_real_calls=4,synthetic_calls=1,pairs=pairs,
            limitation='人工加入/移除两帧的组合干预；不区分单独加帧、去干扰或重复呈现贡献'))
        durable(RUN/'status.json',dict(status='completed',pid=os.getpid(),sid=os.getsid(0),utc=utc()))
        print(pairs,flush=True)


if __name__=='__main__':
    try:execute()
    except BaseException as exc:
        durable(RUN/'failure.json',dict(status='paused',error=repr(exc),utc=utc()));raise
