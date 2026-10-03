"""补查消融完成后的独立计分、同预算子集和输入身份核验，不调用模型。"""
import argparse
import os
os.environ.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable,exclusive,utc,rows
from videoqa_full.report import compare
from videoqa_runtime.baseline_report import wilson
from videoqa_budget32 import development
from videoqa_budget32.run import validate
from videoqa_audit.query_input_probe import BASE,FULL
METHOD='RD-1.2-exp20'
NAME='rd12_exp20__qwen32__dev200__20261003_r1'
RUN=ROOT/'outputs/diagnostics'/NAME
COORD=ROOT/'outputs/methods'/NAME
PARENT=ROOT/'outputs/full_videoqa/rd12_exp12__qwen32__videomme2700__20261003_r1'


def wait_for_completion():
    """接口：只观察已存在协调器；消失或暂停即退出，不重新发起任务。"""
    pid=read_json(COORD/'launcher_identity.json')['pid']
    while True:
        status=read_json(COORD/'status.json')
        if status['status']=='completed':return
        if status['status']=='paused':raise RuntimeError('Coordinator paused')
        process=Path('/proc')/str(pid)/'cmdline'
        if not process.exists():raise RuntimeError('Coordinator absent without completion')
        assert 'scripts/run_coarse_only_ablation.py' in process.read_bytes().decode()
        durable(RUN/'independent_report_status.json',dict(status='waiting',coordinator_pid=pid,pid=os.getpid(),utc=utc()))
        time.sleep(15)


def main(wait=False):
    """接口：只在200条及完成恢复通过后报告，禁止把未完成成绩外推成整组。"""
    with exclusive(RUN/'independent_report.lock'):
        if wait:wait_for_completion()
        assert read_json(COORD/'status.json')['status']=='completed'
        assert read_json(RUN/'completion_audit.json')['status']=='passed'
        spec=development.verify(RUN);data=development.selected_rows(spec)
        sources={str(Path(__file__)):sha256(__file__),str(RUN/'protocol.json'):sha256(RUN/'protocol.json')}
        results={METHOD:{},'RD-1.2-exp12':{},'RD-1.2-exp07':{},'BASE-Uniform':{},'BASE-BLIP-TopK':{}}
        same_rgb=[];mismatches=[];flat=[]
        # 步骤1：重验框架计分并独立用原答案索引复算，源候选严格限定为初始粗池。
        for row in data:
            q=row['question_id'];new=validate(RUN,row,METHOD,spec);poolpath=BASE/'pools'/(q+'.json')
            pool=read_json(poolpath);sources[str(poolpath)]=sha256(poolpath)
            fs=new['selection']['selected_frames'];indices=[f['candidate_index'] for f in fs]
            assert len(fs)==min(32,pool['initial_count']) and all(i<pool['initial_count'] for i in indices)
            assert fs==[pool['candidates'][i] for i in indices]
            paths={METHOD:RUN/'results'/METHOD/(q+'.json'),
                'RD-1.2-exp12':PARENT/'imported/results/RD-1.2-exp12'/(q+'.json'),
                'RD-1.2-exp07':FULL/'imported/results/RD-1.2-exp07'/(q+'.json')}
            paths.update({m:BASE/'results'/m/(q+'.json') for m in ['BASE-Uniform','BASE-BLIP-TopK']})
            for m,p in paths.items():
                r=read_json(p);sources[str(p)]=sha256(p)
                assert r['question']==row['question'] and r['options']==row['options']
                assert r['correct']==(r['answer']['parsed_answer']=='ABCD'[row['answer_index']])
                results[m][q]=r
            parent=results['RD-1.2-exp12'][q]
            if new['rgb_sha256']==parent['rgb_sha256']:
                same_rgb.append(q)
                if new['answer']['parsed_answer']!=parent['answer']['parsed_answer']:mismatches.append(q)
            flat.append(dict(question_id=q,stratum=row['stratum'],new_frames=len(fs),
                parent_frames=len(parent['selection']['selected_frames']),
                parent_fine=sum(f['candidate_index']>=pool['initial_count'] for f in parent['selection']['selected_frames']),
                same_rgb=q in same_rgb,new_correct=new['correct'],parent_correct=parent['correct']))
        # 步骤2：分母事前固定200/196；每个子集独立配对，不用短池差异冒充同预算收益。
        groups={'all200':data,'common32_196':[r for r in data if len(results[METHOD][r['question_id']]['selection']['selected_frames'])==32]}
        assert len(groups['common32_196'])==196
        metrics={};comparisons={}
        for name,group in groups.items():
            metrics[name]={}
            for m,values in results.items():
                metrics[name][m]={}
                for layer in ['all','short','medium','long']:
                    subset=[r for r in group if layer=='all' or r['stratum']==layer]
                    correct=sum(values[r['question_id']]['correct'] for r in subset)
                    metrics[name][m][layer]=dict(n=len(subset),correct=correct,accuracy=correct/len(subset),wilson_95=wilson(correct,len(subset)))
            comparisons[name]={m:compare(group,results[METHOD],values) for m,values in results.items() if m!=METHOD}
        # 步骤3：保留所有得失，后续画面审核按冻结顺序分时长取首个改善/退化，不选有利故事。
        cases=[];pair=comparisons['all200']['RD-1.2-exp12']['cells']
        for layer in ['short','medium','long']:
            for direction,key in [('gain','left_only'),('loss','right_only')]:
                q=next((r['question_id'] for r in data if r['stratum']==layer and r['question_id'] in pair[key]),None)
                if q:cases.append(dict(question_id=q,stratum=layer,direction=direction))
        result=dict(sources=sources,metrics=metrics,comparisons=comparisons,per_question=flat,
            same_rgb=same_rgb,same_rgb_prediction_mismatches=mismatches,case_manifest=cases,
            new_qa_calls=len(list((RUN/'calls'/METHOD).glob('*.json'))),new_model_calls_for_analysis=0,
            limitation='Refinement removal changes local prefixes and quotas together; development set repeatedly observed; not independent raw-video E2E')
        assert result['new_qa_calls']==200
        output=RUN/'ablation_report.json'
        if output.exists():assert read_json(output)==result
        else:durable(output,result)
        durable(RUN/'independent_report_status.json',dict(status='passed',questions=200,utc=utc()))
        print(metrics['all200'],flush=True);print({k:v for k,v in comparisons['all200']['RD-1.2-exp12'].items()},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');a=p.parse_args()
    try:main(a.wait)
    except BaseException as exc:
        durable(RUN/'independent_report_status.json',dict(status='paused',error=repr(exc),utc=utc()));raise
