"""同一exp12方法的M16/K32与M32/K32配对汇总，配置作为独立比较维度。"""
import os
os.environ.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
from pathlib import Path
import sys
import argparse
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable,exclusive,utc
from videoqa_full.report import compare
from videoqa_runtime.baseline_report import wilson
from videoqa_budget32 import development
from videoqa_budget32.run import validate
from videoqa_audit.query_input_probe import BASE,FULL
NAME='rd12_exp12__m16k32__qwen32__dev200__20261003_r1'
METHOD='RD-1.2-exp12'
RUN=ROOT/'outputs/diagnostics'/NAME
COORD=ROOT/'outputs/methods'/NAME
PARENT=ROOT/'outputs/full_videoqa/rd12_exp12__qwen32__videomme2700__20261003_r1'
SOURCE=ROOT/'outputs/analysis/seed16_exp12_200_20261003_r1'


def main(wait=False):
    """接口：全部结果及恢复通过后才计分，不生成新答案，不按分数重试。"""
    with exclusive(RUN/'configuration_report.lock'):
        # 步骤1：等待的是已存在协调器；失踪或暂停即退出，不能启动替代任务。
        pid=read_json(COORD/'launcher_identity.json')['pid']
        while read_json(COORD/'status.json')['status']!='completed':
            if not wait or read_json(COORD/'status.json')['status']=='paused':raise RuntimeError('Coordinator not completed')
            command=Path('/proc')/str(pid)/'cmdline'
            if not command.exists():raise RuntimeError('Coordinator absent')
            assert 'run_seed16_configuration.py' in command.read_bytes().decode()
            durable(RUN/'configuration_report_status.json',dict(status='waiting',coordinator_pid=pid,pid=os.getpid(),utc=utc()))
            time.sleep(15)
        assert read_json(RUN/'completion_audit.json')['status']=='passed'
        spec=development.verify(RUN);assert spec['configuration'].startswith('M16/K32; ')
        data=development.selected_rows(spec);new={};references={k:{} for k in ['exp12_M32_K32','exp07_M32_K32','BASE-Uniform','BASE-BLIP-TopK']}
        sources={str(Path(__file__)):sha256(__file__),str(RUN/'protocol.json'):sha256(RUN/'protocol.json')}
        same=[];mismatches=[];flat=[]
        # 步骤2：校验新池映射与实际源帧，再逐题独立复算原答案索引。
        for row in data:
            q=row['question_id'];r=validate(RUN,row,METHOD,spec);new[q]=r
            source_path=SOURCE/'questions'/(q+'.json');s=read_json(source_path);sources[str(source_path)]=sha256(source_path)
            assert r['selection']['selected_frames']==s['selected_frames'] and len(s['selected_frames'])==32
            paths={'exp12_M32_K32':PARENT/'imported/results'/METHOD/(q+'.json'),
                'exp07_M32_K32':FULL/'imported/results/RD-1.2-exp07'/(q+'.json')}
            paths.update({m:BASE/'results'/m/(q+'.json') for m in ['BASE-Uniform','BASE-BLIP-TopK']})
            for name,p in paths.items():
                value=read_json(p);sources[str(p)]=sha256(p);references[name][q]=value
                assert value['question']==row['question'] and value['options']==row['options']
                assert value['correct']==(value['answer']['parsed_answer']=='ABCD'[row['answer_index']])
            assert r['correct']==(r['answer']['parsed_answer']=='ABCD'[row['answer_index']])
            parent=references['exp12_M32_K32'][q];assert len(parent['selection']['selected_frames'])==32
            if r['rgb_sha256']==parent['rgb_sha256']:
                same.append(q)
                if r['answer']['parsed_answer']!=parent['answer']['parsed_answer']:mismatches.append(q)
            flat.append(dict(question_id=q,stratum=row['stratum'],new_correct=r['correct'],parent_correct=parent['correct'],same_rgb=q in same))
        # 步骤3：标签区分配置，不能把两个相同method ID写进字典而覆盖父结果。
        all_results={'exp12_M16_K32':new,**references};metrics={}
        for name,values in all_results.items():
            metrics[name]={}
            for layer in ['all','short','medium','long']:
                group=[r for r in data if layer=='all' or r['stratum']==layer]
                correct=sum(values[r['question_id']]['correct'] for r in group)
                metrics[name][layer]=dict(n=len(group),correct=correct,accuracy=correct/len(group),wilson_95=wilson(correct,len(group)))
        comparisons={name:compare(data,new,values) for name,values in references.items()}
        cases=[];cells=comparisons['exp12_M32_K32']['cells']
        for layer in ['short','medium','long']:
            for direction,key in [('gain','left_only'),('loss','right_only')]:
                q=next((r['question_id'] for r in data if r['stratum']==layer and r['question_id'] in cells[key]),None)
                if q:cases.append(dict(question_id=q,stratum=layer,direction=direction))
        result=dict(sources=sources,method=METHOD,new_configuration='M16/K32',reference_configuration='M32/K32',
            metrics=metrics,comparisons=comparisons,per_question=flat,same_rgb=same,same_rgb_prediction_mismatches=mismatches,
            case_manifest=cases,new_model_calls_for_analysis=0,scope='Development200 configuration effect; no independent blind-test claim')
        out=RUN/'configuration_report.json'
        if out.exists():assert read_json(out)==result
        else:durable(out,result)
        durable(RUN/'configuration_report_status.json',dict(status='passed',questions=200,utc=utc()))
        print(metrics,flush=True);print(comparisons['exp12_M32_K32'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');a=p.parse_args()
    try:main(a.wait)
    except BaseException as exc:
        durable(RUN/'configuration_report_status.json',dict(status='paused',error=repr(exc),utc=utc()));raise
