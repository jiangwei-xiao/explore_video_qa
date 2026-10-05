"""独立读盘审计：五策略8帧人口、计分、共享身份和唯一调用闭合。"""
import argparse
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable,rows,utc

METHODS=('BASE-Uniform','BASE-BLIP-TopK','RD-1.2','RD-1.2-exp07','RD-1.2-exp12')


def audit(root):
    """接口：只读正式输出独立计分；审计落盘，不触发模型或恢复生成。"""
    root=Path(root);spec=read_json(root/'protocol.json');fp=sha256(root/'protocol.json')
    data=rows();ids={r['question_id'] for r in data};scores={m:0 for m in METHODS};expected=set();call_ids=[];shared=0
    # 步骤1：13500方法记录各自完整，按原标准字母复算，无缺题或额外题。
    for method in METHODS:
        if {p.stem for p in (root/'results'/method).glob('*.json')}!=ids:raise ValueError('Result population mismatch')
    for row in data:
        q=row['question_id'];groups=read_json(root/'inputs/groups'/(q+'.json'));gpus=set()
        for group in groups:
            primary=group['representative'];expected.add((primary,q))
            call=read_json(root/'calls'/primary/(q+'.json'));raw=read_json(root/'raw_generations'/primary/(q+'.json'))
            if call['status']!='completed' or raw['generation_state']!={'started':True,'returned':True}:
                raise ValueError('Uncertain return')
            call_ids.append(call['call_id'])
            if raw['identity']!=dict(question_id=q,method=primary,protocol_sha256=fp,gpu=call['gpu']):
                raise ValueError('Raw identity differs')
            for method in group['methods']:
                r=read_json(root/'results'/method/(q+'.json'));item=read_json(root/'selections'/method/(q+'.json'))
                if r['question']!=row['question'] or r['options']!=row['options'] or r['protocol_sha256']!=fp:
                    raise ValueError('Question/protocol differs')
                if r['selection']!=item['selection'] or r['rgb_sha256']!=item['rgb_sha256'] or item['input_identity']!=group['identity']:
                    raise ValueError('Shared source or input differs')
                a=r['answer'];o=a['observed'];n=len(r['selection']['selected_frames'])
                if not 1<=n<=8 or o['visual_tokens']!=n*210 or o['generate_calls']!=o['prefill_calls'] or o['generate_calls']!=1:
                    raise ValueError('Visual/prefill differs')
                if not 1<=len(o['generated_token_ids'])<=16 or o['max_new_tokens']!=16:
                    raise ValueError('Generation cap differs')
                if o['pixel_sha256']!=group['identity']['pixel_sha256'] or o['input_token_ids']!=group['identity']['input_token_ids']:
                    raise ValueError('Actual input differs')
                if a!=call['answer'] or r['call_id']!=call['call_id'] or r['representative']!=primary or r['gpu']!=call['gpu']:
                    raise ValueError('Shared call differs')
                score=a['parsed_answer']=='ABCD'[row['answer_index']]
                if r['correct']!=score:raise ValueError('Independent score differs')
                if method!=primary and r['timings']:raise ValueError('Shared answer has fabricated physical cost')
                scores[method]+=score;shared+=method!=primary;gpus.add(r['gpu'])
        if len(gpus)!=1:raise ValueError('Question GPU differs')
    # 步骤2：全部调用人口等于冻结U，同题共享不能藏额外生成或遗漏原始返回。
    for folder in ['calls','raw_generations']:
        actual={(p.parent.name,p.stem) for p in (root/folder).glob('*/*.json')}
        if actual!=expected:raise ValueError('Unique input population differs')
    if len(call_ids)!=len(set(call_ids)) or len(call_ids)!=spec['max_calls'] or shared+len(call_ids)!=13500:
        raise ValueError('Call budget differs')
    summary=read_json(root/'summary.json')
    if any(summary['metrics'][m]['all2700']['correct']!=scores[m] for m in METHODS):raise ValueError('Reported scores differ')
    completed=read_json(root/'completion_audit.json')
    if completed['status']!='passed' or completed['extra_generations']!=0:raise ValueError('Completed recovery failed')
    synthetic=read_json(root/'synthetic/acceptance.json')['synthetic_generations']
    synthetic+=sum(read_json(p)['synthetic_generations'] for p in (root/'workers').glob('*.json'))
    result=dict(status='passed',method_results=13500,new_calls=len(call_ids),shared_results=shared,
        scores=scores,synthetic_generations=synthetic,worker_failures=len(list((root/'failures').glob('*.json'))),
        parse_failures={m:len(summary['parse_failures'][m]) for m in METHODS},extra_generations=0,utc=utc())
    durable(root/'independent_completion_audit.json',result);print(result,flush=True)


def main():
    """接口：watch只观察后台状态；完成后自动审计，暂停时不重启实验。"""
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--watch',action='store_true')
    args=parser.parse_args();root=Path(args.run)
    # 步骤1：独立后台观察者允许会话断线，等待不产生任何模型调用。
    if args.watch:
        while True:
            status=read_json(root/'pipeline_status.json')
            if status['status']=='completed':break
            if status['status']=='paused':raise RuntimeError('Pipeline paused; no automatic restart')
            time.sleep(15)
    audit(root)


if __name__=='__main__':main()
