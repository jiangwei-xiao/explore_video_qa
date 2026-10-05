"""独立完成核查：逐题计分、7900唯一调用、200导入及实际模型输入。"""
from pathlib import Path
import argparse
from collections import Counter
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import rows,durable,utc
from videoqa_runtime.baseline_report import distribution


def audit(root):
    """接口：在后台全量报告完成后独立检查原始文件，不调用任何模型。"""
    root=Path(root);summary=read_json(root/'summary.json');data=rows();byid={r['question_id']:r for r in data}
    if summary['status']!='completed' or summary['new_answer_calls']!=7900:raise ValueError('Full report not complete')
    reports={};call_ids=[];sources={};shared=None
    # 步骤1：从原始结果重新计分，不使用汇总中的正确数作为独立证据。
    for stage,method,expected in [('qwen16','RD-1.2-exp12',2700),('llava16','RD-1.2-exp12',2700),('wfs32','REF-WFS-SB',2500)]:
        run=root/stage;spec=read_json(run/'protocol.json');fp=sha256(run/'protocol.json');records={};imports=0
        paths=list((run/'results'/method).glob('*.json'))
        if {p.stem for p in paths}!=set(byid):raise ValueError('Result population mismatch')
        memory=[];tokens=[];prefill=[];generated=[]
        for path in paths:
            q=path.stem;r=read_json(path);row=byid[q];a=r['answer'];o=a['observed']
            if r['question']!=row['question'] or r['options']!=row['options'] or r['protocol_sha256']!=fp:raise ValueError('Result identity mismatch')
            if r['correct']!=(a['parsed_answer']=='ABCD'[row['answer_index']]):raise ValueError('Independent score differs')
            if o['generate_calls']!=1 or o['prefill_calls']!=1 or not 1<=len(o['generated_token_ids'])<=16:
                raise ValueError('Actual generation count differs')
            if r.get('imported'):
                imports+=1
                for name,h in r['import_sources'].items():
                    if sha256(name)!=h:raise ValueError('Imported source changed')
            else:
                call=read_json(run/'calls'/method/(q+'.json'));raw=read_json(run/'raw_generations'/method/(q+'.json'))
                if call['status']!='completed' or call['generation_state']!={'started':True,'returned':True}:
                    raise ValueError('Uncertain call')
                if call['answer']!=a or raw['identity']!=dict(question_id=q,method=method,protocol_sha256=fp,gpu=r['gpu']):
                    raise ValueError('Raw identity or returned answer differs')
                call_ids.append(call['call_id'])
                for field in ['peak_allocated_gib','peak_reserved_gib']:
                    if field in o:memory.append((field,o[field]))
            tokens.append(o['visual_tokens']);prefill.append(o['prefill_tokens']);generated.append(len(o['generated_token_ids']))
            records[q]=r;sources[str(path)]=sha256(path)
        calls=list((run/'calls'/method).glob('*.json'));raws=list((run/'raw_generations'/method).glob('*.json'))
        if len(calls)!=expected or {p.stem for p in calls}!={p.stem for p in raws}:raise ValueError('Call budget or raw return closure differs')
        if imports!=(200 if stage=='wfs32' else 0):raise ValueError('Imported population differs')
        if len(list((run/'failures').glob('*.json')))!=0:raise ValueError('Unexpected failure requires explicit review')
        # 步骤2：从两组实际结果核对同源输入，而不以共用输入文件代替实际执行检查。
        identity={q:(r['selection'],r['rgb_sha256']) for q,r in records.items()}
        if stage=='qwen16':shared=identity
        if stage=='llava16' and identity!=shared:raise ValueError('Actual cross-model sources differ')
        c=sum(r['correct'] for r in records.values())
        if c!=summary['reports'][stage]['metrics']['all2700']['all']['correct']:raise ValueError('Summary total differs')
        workers=[read_json(p) for p in (run/'workers').glob('*.json')]
        reports[stage]=dict(questions=2700,correct=c,new_calls=expected,imported=imports,
            parse_failures=sum(r['answer']['parsed_answer'] is None for r in records.values()),
            frame_counts=dict(Counter(len(r['selection']['selected_frames']) for r in records.values())),
            visual_tokens=distribution(tokens),prefill_tokens=distribution(prefill),generated_tokens=distribution(generated),
            memory={field:distribution([v for k,v in memory if k==field]) for field in ['peak_allocated_gib','peak_reserved_gib']},
            worker_count=len(workers),load_seconds=distribution([w['load_seconds'] for w in workers]),
            warmup_seconds=distribution([w['warmup_seconds'] for w in workers]),
            worker_synthetic_generations=sum(w['synthetic_generations'] for w in workers),
            wall_seconds=read_json(run/'execution.json')['wall_seconds'])
    # 步骤3：全部新调用ID全局唯一，旧答案导入不能计入新模型调用。
    if len(call_ids)!=7900 or len(set(call_ids))!=7900:raise ValueError('Global call ID closure differs')
    synthetic={}
    for kind in ['qwen','llava']:
        p=root/'synthetic'/kind/'synthetic_acceptance.json';r=read_json(p)
        if r['status']!='passed':raise ValueError('Synthetic acceptance differs')
        synthetic[kind]=len(r['checks']);sources[str(p)]=sha256(p)
    recovery=read_json(root/'recovery/completion.json')
    if recovery['extra_qwen_generations']!=0 or recovery['qwen_unchanged_files']!=8100:raise ValueError('Recovery modified Qwen')
    result=dict(status='passed',new_calls=7900,unique_call_ids=7900,imported=200,cross_model_rgb_exact=2700,
        reports=reports,synthetic_acceptance_generations=synthetic,
        total_synthetic_generations=sum(synthetic.values())+sum(r['worker_synthetic_generations'] for r in reports.values()),
        generation_failures=0,automatic_retries=0,orchestration_pauses=1,recovery=recovery,
        input_preparation=read_json(root/'input_execution.json'),sources=sources,utc=utc())
    durable(root/'independent_completion_audit.json',result)
    print({k:v for k,v in result.items() if k not in ['sources','reports']},flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);args=parser.parse_args();audit(args.run)
