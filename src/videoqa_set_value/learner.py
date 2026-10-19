"""冻结BLIP表示上的轻量集合价值头；只学习同题正确/错误输入之间的偏好。"""
from pathlib import Path
import os
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable,exclusive,utc
from videoqa_set_value.data import METHODS


class SetValueHead(nn.Module):
    """接口：B×K×770的冻结帧表示→B个集合分数；不接收答案、选项或方法ID。"""
    def __init__(self):
        super().__init__()
        self.phi=nn.Sequential(nn.Linear(770,128),nn.GELU(),nn.Linear(128,64),nn.GELU())
        self.readout=nn.Linear(128,1)

    def forward(self,features,mask=None):
        """步骤1共享帧网络；步骤2均值/最大值池化；步骤3输出未校准的排序分数。"""
        if features.ndim!=3 or features.shape[-1]!=770:raise ValueError('Expected B,K,770 frozen features')
        if mask is None:mask=torch.ones(features.shape[:2],dtype=torch.bool,device=features.device)
        if mask.shape!=features.shape[:2] or torch.any(mask.sum(1)==0):raise ValueError('Empty or malformed set mask')
        h=self.phi(features);mean=(h*mask[:,:,None]).sum(1)/mask.sum(1)[:,None]
        maximum=h.masked_fill(~mask[:,:,None],-torch.inf).max(1).values
        return self.readout(torch.cat([mean,maximum],dim=-1)).squeeze(-1)


def extract_set_features(fused,probabilities,times,duration):
    """接口：仅图文CLS、ITM及真实位置→K×770；不使用任务答案或输出文字。"""
    z=np.asarray(fused,dtype=np.float32);r=np.asarray(probabilities,dtype=np.float32);t=np.asarray(times,dtype=np.float32)
    if z.ndim!=2 or z.shape[1]!=768 or r.shape!=z.shape[:1] or t.shape!=r.shape or not duration>0:
        raise ValueError('Invalid frozen representation dimensions')
    norms=np.linalg.norm(z,axis=1)
    if not np.isfinite(z).all() or np.any(norms<=1e-12) or not np.isfinite(r).all() or np.any((r<0)|(r>1)):
        raise ValueError('Nonfinite representation or invalid ITM probability')
    positions=t/np.float32(duration)
    if not np.isfinite(positions).all() or np.any((positions<0)|(positions>1)):raise ValueError('Invalid real source position')
    return np.concatenate([z/norms[:,None],r[:,None],positions[:,None]],axis=1).astype(np.float32)


def preference_loss(scores,labels,representatives):
    """接口：同题所有正负对的softplus，先题内平均再题间平均；同输入不重复投票。"""
    per_question=[]
    for s,y,indices in zip(scores,labels,representatives):
        ids=torch.as_tensor(indices,device=s.device,dtype=torch.long);values=s[ids];truth=y[ids]
        positive=values[truth==1];negative=values[truth==0]
        if len(positive) and len(negative):per_question.append(F.softplus(-(positive[:,None]-negative[None,:])).mean())
    if not per_question:raise ValueError('No within-question correctness differences')
    return torch.stack(per_question).mean()


def build_preference_labels(root,rows):
    """接口：只读取训练问答的后验正确性，验证集结果不参与训练数据构造。"""
    root=Path(root);features=[];labels=[];representatives=[];ids=[];all_counts=np.zeros(4,dtype=int)
    for row in rows:
        if row['split']!='train':raise ValueError('Validation labels are forbidden in training')
        q=row['question_id'];plan=read_json(root/'inputs'/(q+'.json'))
        archive=np.load(root/'value_features'/(q+'.npz'),allow_pickle=False)
        x=np.stack([archive[m] for m in METHODS]).astype(np.float32)
        answer=[read_json(root/'qa/train/results'/m/(q+'.json')) for m in METHODS]
        y=np.asarray([int(a['correct']) for a in answer],dtype=np.float32);all_counts+=y.astype(int)
        reps=[group['representative_method_index'] for group in plan['groups']]
        features.append(x);labels.append(y);representatives.append(reps);ids.append(q)
    return np.stack(features),np.stack(labels),representatives,ids,all_counts


def _atomic_torch(path,value):
    """接口：只保存小头/优化器，不包含冻结主模型；完成epoch后原子替换检查点。"""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.part')
    torch.save(value,temp);os.replace(temp,path)


def fit(root):
    """接口：固定600题、20epoch一次训练；恢复只从已落盘完整epoch接续。"""
    root=Path(root)
    with exclusive(root/'fit.lock'):
        manifest=read_json(root/'manifest.json');rows=[r for r in manifest['rows'] if r['split']=='train']
        if len(rows)!=600:raise ValueError('Expected 600 independent training questions')
        identity=dict(manifest=sha256(root/'manifest.json'),inputs=sha256(root/'inputs_complete.json'),
            train_answers=sha256(root/'qa/train/completion_audit.json'),code=sha256(__file__),
            train_features={r['question_id']:sha256(root/'value_features'/(r['question_id']+'.npz')) for r in rows})
        dest=root/'head/checkpoint.pt';summary=root/'head/training.json'
        if summary.exists():
            old=read_json(summary)
            if old['identity']!=identity:raise ValueError('Completed training identity changed')
            if old['status']=='trained' and old.get('checkpoint_sha256')!=sha256(dest):raise ValueError('Completed trained head changed')
            return old
        if list((root/'qa/val/calls').glob('*/*.json')):raise ValueError('Cannot fit after validation generation')
        torch.set_num_threads(2);torch.manual_seed(2027);torch.use_deterministic_algorithms(True)
        x,y,reps,qids,counts=build_preference_labels(root,rows)
        usable=[i for i in range(len(y)) if len(set(y[i,j] for j in reps[i]))==2]
        if not usable:
            outcome=dict(status='no_preference_signal',identity=identity,train_questions=600,usable_questions=0,utc=utc())
            durable(summary,outcome);return outcome
        model=SetValueHead().float();optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
        history=[];start_epoch=0;began=time.perf_counter()
        if dest.exists():
            saved=torch.load(dest,map_location='cpu',weights_only=False)
            if saved['identity']!=identity:raise ValueError('Training checkpoint identity changed')
            model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
            start_epoch=saved['epoch'];history=saved['history']
        features=torch.from_numpy(x);labels=torch.from_numpy(y)
        # 步骤1：每道有偏好题等权；全对/全错题只保留总体统计，不伪造排序标签。
        for epoch in range(start_epoch,20):
            order=np.random.RandomState(2027+epoch).permutation(usable);losses=[]
            for offset in range(0,len(order),32):
                batch=order[offset:offset+32];xb=features[batch];yb=labels[batch]
                scores=model(xb.reshape(-1,8,770)).reshape(len(batch),4)
                loss=preference_loss(scores,yb,[reps[int(i)] for i in batch])
                optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();losses.append(float(loss.detach()))
            history.append(dict(epoch=epoch+1,loss=float(np.mean(losses))))
            # 步骤2：完整epoch后保存；不根据验证集选择epoch或超参数。
            _atomic_torch(dest,dict(identity=identity,epoch=epoch+1,model=model.state_dict(),optimizer=optimizer.state_dict(),history=history))
            durable(root/'head/progress.json',dict(epoch=epoch+1,total_epochs=20,loss=history[-1]['loss'],utc=utc()))
        with torch.inference_mode():train_scores=model(features.reshape(-1,8,770)).reshape(len(rows),4).numpy()
        train_preferences=[];train_selected_correct=0
        for i,row in enumerate(rows):
            values=train_scores[i];best=float(values.max());eligible=[j for j,v in enumerate(values) if best-float(v)<=1e-8]
            plan=read_json(root/'inputs'/(row['question_id']+'.json'))
            chosen=min(eligible,key=lambda j:plan['methods'][METHODS[j]]['source_order'])
            train_selected_correct+=int(y[i,chosen]);pairs=[]
            for a in reps[i]:
                for b in reps[i]:
                    if y[i,a]==1 and y[i,b]==0:
                        d=float(values[a]-values[b]);pairs.append(1. if d>1e-8 else .5 if abs(d)<=1e-8 else 0.)
            if pairs:train_preferences.append(float(np.mean(pairs)))
        outcome=dict(status='trained',identity=identity,train_questions=600,usable_questions=len(usable),
            usable_videos=len({rows[i]['video_id'] for i in usable}),parameters=sum(p.numel() for p in model.parameters()),
            fixed_method_correct=dict(zip(METHODS,counts.tolist())),
            training_best_fixed_method=METHODS[int(np.argmax(counts))],epochs=20,batch_questions=32,seed=2027,
            optimizer=dict(name='AdamW',lr=.001,weight_decay=.0001),history=history,
            train_selected_correct=train_selected_correct,train_preference_accuracy=float(np.mean(train_preferences)),
            training_seconds=time.perf_counter()-began,checkpoint_sha256=sha256(dest),utc=utc())
        durable(summary,outcome);return outcome


def rank_sets(features,checkpoint):
    """接口：只接收候选表示和训练检查点，返回分数；不接收验证答案/方法来源。"""
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False);model=SetValueHead().eval()
    if saved['epoch']!=20:raise ValueError('Expected fixed final epoch')
    model.load_state_dict(saved['model'])
    with torch.inference_mode():return model(torch.as_tensor(features,dtype=torch.float32)).numpy()


def predict_validation(root):
    """接口：独立验证200题的选择必须在首个验证问答前冻结，不从对错选择输入。"""
    root=Path(root)
    with exclusive(root/'prediction.lock'):
        checkpoint=root/'head/checkpoint.pt';head=read_json(root/'head/training.json')
        if head['status']!='trained' or sha256(checkpoint)!=head['checkpoint_sha256']:raise ValueError('Valid fixed head required')
        output=root/'validation_predictions.json'
        if output.exists():
            previous=read_json(output)
            if previous['checkpoint_sha256']!=sha256(checkpoint):raise ValueError('Prediction model changed')
            return previous
        if list((root/'qa/val/calls').glob('*/*.json')):raise ValueError('Validation generation already started')
        rows=[r for r in read_json(root/'manifest.json')['rows'] if r['split']=='val'];predictions=[]
        torch.set_num_threads(2)
        for row in rows:
            q=row['question_id'];archive=np.load(root/'value_features'/(q+'.npz'),allow_pickle=False)
            values=rank_sets(np.stack([archive[m] for m in METHODS]),checkpoint)
            plan=read_json(root/'inputs'/(q+'.json'));best=float(np.max(values))
            eligible=[i for i,v in enumerate(values) if best-float(v)<=1e-8]
            chosen=min(eligible,key=lambda i:plan['methods'][METHODS[i]]['source_order'])
            means=[float(archive[m][:,768].mean()) for m in METHODS]
            itm_best=max(means);itm_ids=[i for i,v in enumerate(means) if itm_best-v<=1e-8]
            itm=min(itm_ids,key=lambda i:plan['methods'][METHODS[i]]['source_order'])
            predictions.append(dict(question_id=q,scores=[float(v) for v in values],chosen_method=METHODS[chosen],
                mean_itm_method=METHODS[itm],feature_sha256=sha256(root/'value_features'/(q+'.npz'))))
        if len(predictions)!=200:raise ValueError('Expected 200 frozen validation predictions')
        result=dict(checkpoint_sha256=sha256(checkpoint),training_summary_sha256=sha256(root/'head/training.json'),
            manifest_sha256=sha256(root/'manifest.json'),predictions=predictions,uses_validation_outcomes=False,utc=utc())
        durable(output,result);return result
