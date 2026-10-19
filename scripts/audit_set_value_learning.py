"""独立复核800题轻量学习：原始生成、官方标签、选择/训练/验证隔离与预算。"""
import argparse
from collections import Counter,defaultdict
import csv
import datetime
import json
import os
from pathlib import Path
import sys
import time

os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
ROOT=Path(os.environ.get('VIDEOQA_SET_VALUE_ROOT',str(Path(__file__).resolve().parents[1]))).resolve();sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable,utc

METHODS=('BASE-Uniform','BASE-BLIP-TopK','RD-1.2-exp07','RD-1.2-exp12')


def option(text):
    """接口：独立实现事前严格五字母语法，失败不随机回退。"""
    s=text.strip()
    if s in 'ABCDE' and len(s)==1:return s
    if len(s)==2 and s[0] in 'ABCDE' and s[1]=='.':return s[0]
    if len(s)==3 and s[0]=='(' and s[1] in 'ABCDE' and s[2]==')':return s[1]
    return None


def check(root):
    """接口：只读已完成原始结果；小头CPU重放不是新增教师或骨干模型调用。"""
    root=Path(root);start=time.monotonic();torch.set_num_threads(2)
    manifest=read_json(root/'manifest.json');rows=manifest['rows'];summary=read_json(root/'report.json')
    frozen=read_json(root/'validation_predictions.json');trained=read_json(root/'head/training.json')
    assert len(rows)==800 and len({r['question_id'] for r in rows})==800
    assert Counter(r['split'] for r in rows)=={'train':600,'val':200}
    videos=defaultdict(list)
    for r in rows:videos[r['video_id']].append(r)
    assert len(videos)==400 and all(len(rs)==2 and len({r['split'] for r in rs})==1 for rs in videos.values())
    assert Counter(rs[0]['split'] for rs in videos.values())=={'train':300,'val':100}
    assert len({rs[0]['video_sha256'] for rs in videos.values()})==400
    official={};ambiguities=[];counts={s:dict.fromkeys(METHODS,0) for s in ('train','val')}
    for split in ('train','val'):
        with (root/'official'/(split+'.csv')).open(newline='') as f:official[split]=list(csv.DictReader(f))
    tokenizer=AutoTokenizer.from_pretrained('/home/models/lmms-lab/LLaVA-Video-7B-Qwen2',local_files_only=True)
    raw_calls={};artifacts={};correct={};plans={};gpu_by_video=defaultdict(set)
    # 步骤1：从官方源行和原始Token独立复算，不把结果中的correct当真值。
    for r in rows:
        q=r['question_id'];source=official[r['split']][r['official_row']]
        assert (source['video'],source['type'],source['qid'],source['question'])==(r['video_id'],r['task_type'],r['official_qid'],r['question'])
        assert [source['a'+str(i)] for i in range(5)]==r['options'] and int(source['answer'])==r['answer_index']
        if r['options'].count(r['options'][r['answer_index']])>1:ambiguities.append(q)
        plan=read_json(root/'inputs'/(q+'.json'));plans[q]=plan
        assert plan['split']==r['split'] and plan['uses_answer_for_partition'] is False
        for m in METHODS:
            item=read_json(root/'qa'/r['split']/'results'/m/(q+'.json'));path=Path(item['call_path'])
            assert path.parent.parent.parent==root/'qa'/r['split'] and path.stem==q
            call=read_json(path);rawpath=path.parent.parent.parent/'raw_generations'/path.parent.name/path.name
            raw=read_json(rawpath);assert call['status']=='completed'
            assert raw['generation_state']=={'started':True,'returned':True}
            assert raw['identity']==dict(question_id=q,method=path.parent.name,gpu=call['gpu'],protocol_sha256=call['protocol_sha256'])
            obs=raw['observed'];tokens=obs['generated_token_ids'];assert 1<=len(tokens)<=16
            text=tokenizer.decode(tokens,skip_special_tokens=True).strip()
            assert text==raw['raw_generated_text'].strip()==item['answer']['raw_output']==call['answer']['raw_output']
            y=option(text)=='ABCDE'[r['answer_index']]
            assert y==item['correct'] and option(text)==item['answer']['parsed_answer']
            assert item['input_key']==plan['methods'][m]['input_key']
            assert obs['visual_tokens']==1680 and obs['source_frames']==8 and obs['max_new_tokens']==16
            for k in ('prompt','input_token_ids','pixel_sha256'):
                assert obs[k]==plan['methods'][m]['identity'][k]
            assert call['protocol_sha256']==sha256(root/'qa'/r['split']/'protocol.json')
            correct[(q,m)]=int(y);counts[r['split']][m]+=int(y);gpu_by_video[r['video_id']].add(call['gpu'])
            raw_calls[str(path)]=call
            for p in (path,rawpath,root/'qa'/r['split']/'results'/m/(q+'.json')):artifacts[str(p.relative_to(root))]=sha256(p)
    assert all(len(gs)==1 for gs in gpu_by_video.values())
    physical={s:sum(Path(p).parent.parent.parent.name==s for p in raw_calls) for s in ('train','val')}
    budget=read_json(root/'inputs_complete.json')['real_call_budgets'];assert physical==budget and sum(physical.values())<=3200
    assert len({c['call_id'] for c in raw_calls.values()})==len(raw_calls)
    synthetic=list((root/'qa/train/calls/synthetic').glob('*.json'));assert len(synthetic)<=8
    assert sum(len(list((root/'qa'/s/'calls'/m).glob('*.json'))) for s in ('train','val') for m in METHODS)==len(raw_calls)
    # 步骤2：验证冻结时间和检查点链，验证QA不能早于预测发布。
    assert frozen['checkpoint_sha256']==trained['checkpoint_sha256']==sha256(root/'head/checkpoint.pt')
    assert frozen['training_summary_sha256']==sha256(root/'head/training.json')
    assert trained['identity']['train_answers']==sha256(root/'qa/train/completion_audit.json')
    predictions={p['question_id']:p for p in frozen['predictions']};assert len(predictions)==200
    val_calls=[c for c in raw_calls.values() if c['question_id'] in predictions]
    earliest=min((c['started_at'] for c in val_calls),key=datetime.datetime.fromisoformat)
    assert datetime.datetime.fromisoformat(frozen['utc'])<datetime.datetime.fromisoformat(earliest)
    for r in rows:
        if r['split']=='train':assert trained['identity']['train_features'][r['question_id']]==sha256(root/'value_features'/(r['question_id']+'.npz'))
    saved=torch.load(root/'head/checkpoint.pt',map_location='cpu',weights_only=False)
    assert saved['epoch']==20 and len(saved['history'])==20 and [x['epoch'] for x in saved['history']]==list(range(1,21))
    state=saved['model'];assert sum(v.numel() for v in state.values())==107073
    assert all(v.dtype==torch.float32 for v in state.values())
    assert saved['optimizer']['param_groups'][0]['lr']==.001 and saved['optimizer']['param_groups'][0]['weight_decay']==.0001
    # 步骤3：直接从张量运算重放小头，不调用训练模块或教师生成。
    learned=0;means=0;preference=[];maximum_error=0.
    for r in rows:
        if r['split']!='val':continue
        q=r['question_id'];p=predictions[q];path=root/'value_features'/(q+'.npz')
        assert sha256(path)==p['feature_sha256']
        with np.load(path,allow_pickle=False) as arrays:x=torch.from_numpy(np.stack([arrays[m] for m in METHODS]))
        assert x.shape==(4,8,770) and torch.isfinite(x).all()
        assert torch.allclose(x[:,:,:768].norm(dim=-1),torch.ones((4,8)),atol=1e-5,rtol=0)
        assert bool(((x[:,:,768:]>=0)&(x[:,:,768:]<=1)).all())
        with torch.inference_mode():
            h=F.gelu(F.linear(x,state['phi.0.weight'],state['phi.0.bias']))
            h=F.gelu(F.linear(h,state['phi.2.weight'],state['phi.2.bias']))
            scores=F.linear(torch.cat([h.mean(1),h.max(1).values],1),state['readout.weight'],state['readout.bias']).flatten().numpy()
        maximum_error=max(maximum_error,float(np.max(np.abs(scores-np.asarray(p['scores'])))));assert maximum_error<=1e-6
        best=max(p['scores']);chosen=min((i for i,s in enumerate(p['scores']) if best-s<=1e-8),key=lambda i:plans[q]['methods'][METHODS[i]]['source_order'])
        assert METHODS[chosen]==p['chosen_method'];learned+=correct[(q,p['chosen_method'])];means+=correct[(q,p['mean_itm_method'])]
        reps=[g['representative_method_index'] for g in plans[q]['groups']];pairs=[]
        for a in reps:
            for b in reps:
                if correct[(q,METHODS[a])]==1 and correct[(q,METHODS[b])]==0:
                    d=p['scores'][a]-p['scores'][b];pairs.append(1. if d>1e-8 else .5 if abs(d)<=1e-8 else 0.)
        if pairs:preference.append(float(np.mean(pairs)))
    assert counts['train']==trained['fixed_method_correct']
    assert counts['val']=={m:summary['counts'][m] for m in METHODS}
    assert learned==summary['counts']['learned_selection'] and means==summary['counts']['mean_itm_selection']
    assert len(preference)==summary['preference_questions']
    assert not preference or abs(float(np.mean(preference))-summary['preference_accuracy'])<1e-12
    return dict(status='passed',questions=800,method_results=3200,video_clusters=400,
        train_validation_video_overlap=0,physical_real_calls=physical,synthetic_calls=len(synthetic),
        independent_correct=counts,learned_validation_correct=learned,mean_itm_validation_correct=means,
        preference_questions=len(preference),maximum_head_replay_abs_error=maximum_error,
        validation_prediction_utc=frozen['utc'],earliest_validation_call_utc=earliest,
        official_duplicate_correct_option_questions=ambiguities,artifacts=artifacts,
        new_teacher_generations=0,new_backbone_forwards=0,auditor_sha256=sha256(__file__),audit_seconds=time.monotonic()-start,utc=utc())


def main():
    """接口：审计仅能用于已完成本轮目录；失败保留异常，不补发生成。"""
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args()
    if not a.run_id.replace('_','').isalnum():p.error('Invalid run ID')
    root=ROOT/'outputs/set_value'/a.run_id
    result=check(root);durable(root/'independent_completion_audit.json',result)
    print({k:v for k,v in result.items() if k!='artifacts'})


if __name__=='__main__':main()
