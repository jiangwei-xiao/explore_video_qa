"""独立复算exp07双模型全量结果、3806调用、1594导入及实际输入身份。"""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import rows,durable,utc
from videoqa_generalization.exp07_budget16 import METHOD,SOURCE


def audit(root):
    """接口：只读取已完成结果，独立按标准字母计分，源答案/CPU验收共同核验。"""
    root=Path(root);summary=read_json(root/'summary.json');partition=read_json(root/'input_summary.json')
    same=set(partition['imported_ids']);new=set(partition['new_ids']);data=rows();ids={r['question_id'] for r in data}
    if len(same)!=797 or len(new)!=1903 or same&new or same|new!=ids:raise ValueError('Partition differs')
    calls=[];groups={};details={};files={}
    # 步骤1：从5400个原始结果独立计分，逐题源输入和原始返回不能缺失。
    for stage in ['qwen16','llava16']:
        run=root/stage;spec=read_json(run/'protocol.json');fp=sha256(run/'protocol.json');group={};correct=0
        if {p.stem for p in (run/'results'/METHOD).glob('*.json')}!=ids:raise ValueError('Result closure differs')
        for row in data:
            q=row['question_id'];p=run/'results'/METHOD/(q+'.json');r=read_json(p);item=read_json(run/'selections'/METHOD/(q+'.json'))
            if r['question']!=row['question'] or r['options']!=row['options'] or r['protocol_sha256']!=fp:
                raise ValueError('Question or protocol identity differs')
            a=r['answer'];o=a['observed'];score=a['parsed_answer']=='ABCD'[row['answer_index']]
            if score!=r['correct']:raise ValueError('Independent score differs')
            if r['selection']!=item['selection'] or r['rgb_sha256']!=item['rgb_sha256'] or len(r['rgb_sha256'])!=16:
                raise ValueError('Actual frame identity differs')
            if o['generate_calls']!=1 or o['prefill_calls']!=1 or not 1<=len(o['generated_token_ids'])<=16:
                raise ValueError('Generation observation differs')
            if q in same:
                if not r.get('imported'):raise ValueError('Expected imported result')
                old=read_json(SOURCE/stage/'results/RD-1.2-exp12'/(q+'.json'))
                if r['answer']!=old['answer'] or r['selection']!=old['selection'] or r['rgb_sha256']!=old['rgb_sha256']:
                    raise ValueError('Imported answer or input changed')
                for name,h in r['import_sources'].items():
                    if sha256(name)!=h:raise ValueError('Imported origin changed')
                if sha256(r['import_check'])!=r['import_check_sha256']:raise ValueError('CPU acceptance changed')
            else:
                if r.get('imported'):raise ValueError('Changed input was imported')
                call=read_json(run/'calls'/METHOD/(q+'.json'));raw=read_json(run/'raw_generations'/METHOD/(q+'.json'))
                if call['status']!='completed' or call['answer']!=a or call['generation_state']!={'started':True,'returned':True}:
                    raise ValueError('Uncertain or different answer')
                if raw['identity']!=dict(question_id=q,method=METHOD,protocol_sha256=fp,gpu=r['gpu']):raise ValueError('Raw return identity differs')
                calls.append(call['call_id'])
            correct+=score;group[q]=r;files[str(p)]=sha256(p)
        if {p.stem for p in (run/'calls'/METHOD).glob('*.json')}!=new or {p.stem for p in (run/'raw_generations'/METHOD).glob('*.json')}!=new:
            raise ValueError('Generated population differs')
        if list((run/'failures').glob('*.json')):raise ValueError('Failure requires review')
        if correct!=summary['reports'][stage]['metrics']['all2700']['all']['correct']:raise ValueError('Summary score differs')
        details[stage]=dict(questions=2700,correct=correct,new_calls=1903,imported=797,
            parse_failures=sum(r['answer']['parsed_answer'] is None for r in group.values()))
        groups[stage]=group
    # 步骤2：独立读取两模型实际结果，逐题源RGB一致；3806新调用全局唯一。
    for q in ids:
        a,b=groups['qwen16'][q],groups['llava16'][q]
        if a['selection']!=b['selection'] or a['rgb_sha256']!=b['rgb_sha256']:raise ValueError('Cross-model source differs')
    if len(calls)!=3806 or len(set(calls))!=3806:raise ValueError('Call ID budget differs')
    synthetic=0
    for kind in ['qwen','llava']:
        accepted=read_json(root/'synthetic'/kind/'acceptance.json')
        if accepted['status']!='passed':raise ValueError('Synthetic acceptance incomplete')
        synthetic+=accepted['synthetic_generations']
    for stage in ['qwen16','llava16']:
        synthetic+=sum(read_json(p)['synthetic_generations'] for p in (root/stage/'workers').glob('*.json'))
    # 步骤3：完成恢复证据必须真实为零生成，不把导入计作新问答。
    completed=read_json(root/'completion_audit.json')
    if completed['status']!='passed' or completed['unchanged_files']!=13012 or completed['extra_generations']!=0:
        raise ValueError('Completed recovery differs')
    result=dict(status='passed',new_calls=3806,unique_call_ids=3806,imported=1594,cross_model_source_exact=2700,
        synthetic_generations=synthetic,generation_failures=0,retries=0,details=details,files=files,utc=utc())
    durable(root/'independent_completion_audit.json',result)
    print({k:v for k,v in result.items() if k!='files'},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);args=parser.parse_args();audit(args.run)
