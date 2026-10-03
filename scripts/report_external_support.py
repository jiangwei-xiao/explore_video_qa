"""exp21对exp12的完整配对核验，等待已有后台任务，不生成答案。"""
import os
os.environ.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
from pathlib import Path
import sys
import time
import argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable,exclusive,utc
from videoqa_full.report import compare
from videoqa_runtime.baseline_report import wilson
from videoqa_budget32 import development
from videoqa_budget32.run import validate
from videoqa_audit.query_input_probe import BASE,FULL
NAME='rd12_exp21__qwen32__dev200__20261003_r1'
METHOD='RD-1.2-exp21'
RUN=ROOT/'outputs/diagnostics'/NAME
COORD=ROOT/'outputs/methods'/NAME
PARENT=ROOT/'outputs/full_videoqa/rd12_exp12__qwen32__videomme2700__20261003_r1'


def main(wait):
    """接口：200条及600文件恢复通过后统一计分，保留所有得失及同输入不一致项。"""
    with exclusive(RUN/'paired_report.lock'):
        pid=read_json(COORD/'launcher_identity.json')['pid']
        # 步骤1：只等待具体活进程，不因观察超时或锁文件存在而重新启动模型。
        while read_json(COORD/'status.json')['status']!='completed':
            if not wait or read_json(COORD/'status.json')['status']=='paused':raise RuntimeError('Coordinator not completed')
            p=Path('/proc')/str(pid)/'cmdline'
            if not p.exists():raise RuntimeError('Coordinator absent')
            assert 'run_external_support_development.py' in p.read_bytes().decode()
            durable(RUN/'paired_report_status.json',dict(status='waiting',pid=os.getpid(),coordinator_pid=pid,utc=utc()))
            time.sleep(15)
        assert read_json(RUN/'completion_audit.json')['status']=='passed'
        spec=development.verify(RUN);data=development.selected_rows(spec)
        groups={m:{} for m in [METHOD,'RD-1.2-exp12','RD-1.2-exp07','BASE-Uniform','BASE-BLIP-TopK']}
        sources={str(Path(__file__)):sha256(__file__),str(RUN/'protocol.json'):sha256(RUN/'protocol.json')};same=[];mismatches=[]
        # 步骤2：校验实际调用、输入及框架计分，原标准答案索引独立复算。
        for row in data:
            q=row['question_id'];validate(RUN,row,METHOD,spec)
            paths={METHOD:RUN/'results'/METHOD/(q+'.json'),'RD-1.2-exp12':PARENT/'imported/results/RD-1.2-exp12'/(q+'.json'),
                'RD-1.2-exp07':FULL/'imported/results/RD-1.2-exp07'/(q+'.json')}
            paths.update({m:BASE/'results'/m/(q+'.json') for m in ['BASE-Uniform','BASE-BLIP-TopK']})
            for m,p in paths.items():
                r=read_json(p);sources[str(p)]=sha256(p);groups[m][q]=r
                assert r['question']==row['question'] and r['options']==row['options']
                assert r['correct']==(r['answer']['parsed_answer']=='ABCD'[row['answer_index']])
            new,parent=groups[METHOD][q],groups['RD-1.2-exp12'][q]
            assert len(new['selection']['selected_frames'])==len(parent['selection']['selected_frames'])==32
            if new['rgb_sha256']==parent['rgb_sha256']:
                a,b=new['answer']['observed'],parent['answer']['observed']
                assert a['processor_pixel_sha256']==b['processor_pixel_sha256'] and a['input_token_ids']==b['input_token_ids']
                same.append(q)
                if new['answer']['parsed_answer']!=parent['answer']['parsed_answer']:mismatches.append(q)
        # 步骤3：固定全部对照，选择有界案例但不挑选更有利的旧答案。
        metrics={}
        for method,values in groups.items():
            metrics[method]={}
            for layer in ['all','short','medium','long']:
                part=[r for r in data if layer=='all' or r['stratum']==layer];correct=sum(values[r['question_id']]['correct'] for r in part)
                metrics[method][layer]=dict(n=len(part),correct=correct,accuracy=correct/len(part),wilson_95=wilson(correct,len(part)))
        comparisons={m:compare(data,groups[METHOD],v) for m,v in groups.items() if m!=METHOD}
        cases=[];cells=comparisons['RD-1.2-exp12']['cells']
        for layer in ['short','medium','long']:
            for direction,key in [('gain','left_only'),('loss','right_only')]:
                q=next((r['question_id'] for r in data if r['stratum']==layer and r['question_id'] in cells[key]),None)
                if q:cases.append(dict(question_id=q,stratum=layer,direction=direction))
        result=dict(sources=sources,metrics=metrics,comparisons=comparisons,same_rgb=same,same_rgb_prediction_mismatches=mismatches,
            case_manifest=cases,new_model_calls_for_analysis=0,scope='Frozen development200, not blind testing or raw-video E2E')
        path=RUN/'paired_report.json'
        if path.exists():assert read_json(path)==result
        else:durable(path,result)
        durable(RUN/'paired_report_status.json',dict(status='passed',questions=200,utc=utc()))
        print(metrics,flush=True);print(comparisons['RD-1.2-exp12'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');a=p.parse_args()
    try:main(a.wait)
    except BaseException as exc:
        durable(RUN/'paired_report_status.json',dict(status='paused',error=repr(exc),utc=utc()));raise
