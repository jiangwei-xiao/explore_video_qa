"""原问题ITC定位与现有ITM入口对照：200题＋固定8题，最终问答0。"""
import os
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
from pathlib import Path
import sys
import argparse
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
from videoqa_runtime.common import read_json,sha256,offline_environment
from videoqa_full.state import durable,exclusive,save_array,rows,utc
from videoqa_audit.blip_projection import load_projection_parameters,recover_projected_features
from videoqa_audit.blip_itc_entry import rank_itc_candidates,original_question_itc_embedding,score_itc_entry
from videoqa_audit.option_coarse_replay import plan_from_seeds
from videoqa_budget32.selection import config
from analyze_coverage_objectives import BASE,DEV,FACTS

MODEL=Path('/home/models/Salesforce/blip-itm-base-coco')
BANK=ROOT/'outputs/diagnostics/rd12_exp24_visual_states_20261007_r3'
PILOT=('035-2','018-2','363-2','409-3','556-1','383-2','003-2','006-1')


def prepare(root):
    """接口：冻结200题及固定面板、原始问题、投影验证与来源；不重新评分或问答。"""
    root.mkdir(parents=True,exist_ok=True)
    if (root/'protocol.json').exists():return verify(root)
    data={r['question_id']:r for r in rows()}
    ids=list(dict.fromkeys([r['question_id'] for r in read_json(DEV)['rows']]+list(PILOT)))
    source_paths=[DEV,FACTS,MODEL/'pytorch_model.bin',ROOT/'src/videoqa_audit/blip_projection.py',
        ROOT/'src/videoqa_audit/blip_itc_entry.py',Path(__file__),ROOT/'src/videoqa_audit/option_coarse_replay.py',
        ROOT/'src/videoqa_methods/density_selection.py',ROOT/'configs/retrieval_density_v1.json',
        BANK/'protocol.json',ROOT/'outputs/analysis/blip_projection_bank6_20261002_r1/summary.json']
    source_paths.extend(p for p in MODEL.iterdir() if p.suffix in ('.json','.txt'))
    source_paths.extend([ROOT/'src/videoqa_runtime/baseline_selection.py',ROOT/'src/videoqa_full/state.py',
        ROOT/'data/manifests/video_mme_full_2700.json'])
    inputs={}
    # 步骤1：没有图片模型调用；原问题是文本编码器唯一任务输入。
    for q in ids:
        pp,fp=BASE/'pools'/(q+'.json'),BASE/'features'/(q+'.npy')
        source_paths.extend([pp,fp])
        inputs[q]=dict(question=data[q]['question'],pool=str(pp),features=str(fp))
    checked=read_json(ROOT/'outputs/analysis/blip_projection_bank6_20261002_r1/summary.json')
    if checked['status']!='completed' or max(r['cached_projection_error'] for values in checked['question_errors'].values() for r in values)>1e-6:
        raise ValueError('Original image projection recovery not validated')
    import inspect
    from transformers.models.blip.modeling_blip import BlipForImageTextRetrieval
    source_paths.append(Path(inspect.getfile(BlipForImageTextRetrieval)))
    sources={str(p):sha256(p) for p in source_paths}
    from transformers import BlipProcessor
    tokenizer=BlipProcessor.from_pretrained(MODEL,local_files_only=True).tokenizer
    for q,item in inputs.items():
        item['token_ids']=tokenizer(item['question'],truncation=False)['input_ids']
        if len(item['token_ids'])>512:raise ValueError('Original question exceeds BLIP position limit')
    # 步骤2：4张均匀银行图/题是固定官方接口验证，不按输出选择验证样本。
    bank=read_json(BANK/'protocol.json')['banks']
    checks={}
    for q in PILOT:
        ids4=[0,bank[q]['count']//3,2*bank[q]['count']//3,bank[q]['count']-1]
        checks[q]=dict(indices=ids4,old_feature_file=bank[q]['old_feature_file'])
        source_paths.extend([Path(bank[q]['old_feature_file'])])
        for i in ids4:
            source_paths.append(BANK/'bank'/q/f'{i:03d}.png')
    sources={str(p):sha256(p) for p in source_paths}
    spec=dict(inputs=inputs,question_ids=ids,development_ids=[r['question_id'] for r in read_json(DEV)['rows']],
        validation=checks,sources=sources,actual_text_encodings=len(ids),real_image_validation_limit=32,
        synthetic_image_validation_limit=4,qa_calls=0,seed_budget=32,frame_budget=32,
        distinction='Official image-text ITC query similarity for entry, not exp16 image-image projected support',
        scope='Localization proposal comparison only, original ITM remains anchor/support signal; no new refined pool or answer',utc=utc())
    import shutil
    for path in source_paths:
        if str(path).startswith(str(ROOT/'src')) or path==Path(__file__):
            dest=root/'code_snapshot'/path.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
    durable(root/'protocol.json',spec)
    return spec


def verify(root):
    """接口：来源指纹不一致时暂停；不能把旧实验头像参数当作新协议结果。"""
    s=read_json(root/'protocol.json')
    for p,h in s['sources'].items():
        if sha256(p)!=h:raise ValueError('Frozen ITC probe source changed: '+p)
    return s


def text_once(root,q,question,scorer):
    """接口：原题文本前向开始前落盘；已确认返回复用，未知前向禁止重跑。"""
    path=root/'text_calls'/(q+'.json');array=path.with_suffix('.npy')
    identity=dict(question=question,protocol_sha256=sha256(root/'protocol.json'))
    if path.exists():
        record=read_json(path)
        if record['status']!='completed' or record['identity']!=identity or sha256(array)!=record['feature_sha256']:
            raise ValueError('Uncertain text encoding; repetition forbidden')
        return np.load(array,allow_pickle=False)
    durable(path,dict(status='started_may_forward',identity=identity,utc=utc()))
    started=time.perf_counter();feature,tokens=original_question_itc_embedding(scorer,question)
    durable(path,dict(status='completed',identity=identity,feature_sha256=save_array(array,feature),token_ids=tokens,
        seconds=time.perf_counter()-started,text_generation_calls=0,qa_calls=0,utc=utc()))
    return feature


def validate_once(root,q,scorer,images,old_features,text_feature,params):
    """接口：官方端到端ITC与缓存恢复评分对齐，32真实/4合成上限，失败不调整容差。"""
    import torch
    import hashlib
    path=root/'validation_calls'/(q+'.json')
    identity=dict(protocol_sha256=sha256(root/'protocol.json'),rgb_sha256=[hashlib.sha256(np.asarray(im).tobytes()).hexdigest() for im in images])
    if path.exists():
        old=read_json(path)
        if old['status']!='completed' or old['identity']!=identity:raise ValueError('Uncertain official ITC validation')
        return old
    encoded=scorer.tokenize('Which synthetic color is visible?' if q=='synthetic' else read_json(root/'protocol.json')['inputs'][q]['question'])
    pixels=scorer.processor(images=images,return_tensors='pt')['pixel_values'].cuda().float()
    inputs={k:v.repeat(len(images),1).cuda() for k,v in encoded.items() if k in ('input_ids','attention_mask')}
    # 步骤1：真实图像前向只用于官方分支对齐，不从该调用更新全视频评分资产。
    durable(path,dict(status='started_may_forward',identity=identity,source_images=len(images),utc=utc()))
    with torch.inference_mode():
        output=scorer.model(pixel_values=pixels,**inputs,use_itm_head=False)
        actual=output.itm_score.diag().cpu().numpy()
        direct_features=torch.nn.functional.normalize(output.last_hidden_state[:,0,:],dim=-1).cpu().numpy()
        text=torch.nn.functional.normalize(scorer.model.text_proj(output.question_embeds[:,0]),dim=-1).cpu().numpy()[0]
    restored,_=score_itc_entry(old_features if old_features is not None else direct_features,text_feature if text_feature is not None else text,params)
    error=float(np.max(np.abs(restored-actual)))
    # 步骤2：同源纯视觉与文本路径需1e-6对齐，不宣称比较分数是答对概率。
    if error>1e-6:raise ValueError(('Official ITC recovery mismatch',q,error))
    result=dict(status='completed',identity=identity,source_images=len(images),maximum_score_error=error,
        actual=actual.tolist(),restored=restored.tolist(),utc=utc())
    durable(path,result);return result


def window_hit(pool,plan,pts):
    """接口：精确有理数窗口包含关系，已列见证可达不是实际取证或充分性。"""
    from fractions import Fraction
    time=(pts-pool['video']['start_pts'])*Fraction(pool['video']['time_base'])
    return any(Fraction(r['left_fraction'])<=time<=Fraction(r['right_fraction']) for r in plan['regions'])


def analyze(root,spec,params):
    """接口：全200题定位记录与42题75事实，人工目录仅用于评价，不进入种子生成。"""
    records={}
    for n,q in enumerate(spec['question_ids'],1):
        item=spec['inputs'][q];pool=read_json(item['pool']);initial=pool['initial_count']
        features=np.load(item['features'],mmap_mode='r')[:initial]
        text=np.load(root/'text_calls'/(q+'.npy'),allow_pickle=False)
        scores,scales=score_itc_entry(features,text,params)
        seeds=rank_itc_candidates(pool['candidates'][:initial],scores,32)
        plan=plan_from_seeds(pool['candidates'][:initial],pool['initial_scores'],seeds,pool['video'],config(32,32))
        old=pool['plan'];path=root/'scores'/(q+'.npy');save_array(path,scores)
        # 步骤1：原问题ITM分数不覆写，只记录不同入口及其实际时间并集。
        r=dict(question_id=q,seeds=seeds,old_seeds=old['seed_indices'],plan=plan,old_plan=old,
            score_file=str(path),score_sha256=sha256(path),score_min=float(scores.min()),score_max=float(scores.max()),
            recovered_scale_min=float(scales.min()),recovered_scale_max=float(scales.max()),new_refinement_performed=False,qa_calls=0)
        durable(root/'records'/(q+'.json'),r);records[q]=r
        if n%20==0:durable(root/'status.json',dict(status='localization_analysis',completed=n,total=len(spec['question_ids']),utc=utc()))
    facts=[]
    # 步骤2：冻结75事实分母不变，不把没有标注的候选判为无关。
    for fact in read_json(FACTS)['entries']:
        q=fact['question_id'];r=records[q];pool=read_json(spec['inputs'][q]['pool']);positions={c['source_pts']:i for i,c in enumerate(pool['candidates'][:pool['initial_count']])}
        witnesses=[positions[p] for p in fact['witness_pts']]
        facts.append(dict(question_id=q,fact_id=fact['fact_id'],description=fact['description'],
            old_seed=bool(set(witnesses)&set(r['old_seeds'])),new_seed=bool(set(witnesses)&set(r['seeds'])),
            old_window=any(window_hit(pool,r['old_plan'],p) for p in fact['witness_pts']),
            new_window=any(window_hit(pool,r['plan'],p) for p in fact['witness_pts'])))
    qids=sorted({f['question_id'] for f in facts})
    if len(facts)!=75 or len(qids)!=42:raise ValueError('Frozen proxy denominator changed')
    def macro(key):
        return float(np.mean([np.mean([f[key] for f in facts if f['question_id']==q]) for q in qids]))
    summary=dict(status='completed',facts=facts,metrics={k:dict(count=sum(f[k] for f in facts),macro=macro(k)) for k in ('old_seed','new_seed','old_window','new_window')},
        window_gained=[(f['question_id'],f['fact_id']) for f in facts if not f['old_window'] and f['new_window']],
        window_lost=[(f['question_id'],f['fact_id']) for f in facts if f['old_window'] and not f['new_window']],
        exact_seed_changes=sum(set(r['seeds'])!=set(r['old_seeds']) for q,r in records.items() if q in spec['development_ids']),
        actual_text_encodings=len(spec['question_ids']),validation_text_inputs=36,
        real_image_validation=32,synthetic_image_validation=4,
        actual_image_features_computed_in_analysis=0,qa_calls=0,
        limitation='Entry/window proxy only; coarse candidates remain globally eligible in exp12; no final selection/accuracy or semantic sufficiency')
    summary['artifact_hashes']={str(p):sha256(p) for folder in ('text_calls','validation_calls','scores','records') for p in (root/folder).iterdir()}
    durable(root/'summary.json',summary);print(summary['metrics'],flush=True)
    return summary


def execute(root,gpu):
    """接口：单卡只编码原题，官方32帧验证后CPU复用缓存检索；完成恢复0新模型前向。"""
    offline_environment();spec=verify(root)
    if (root/'summary.json').exists():
        saved=read_json(root/'summary.json')
        for p,h in saved['artifact_hashes'].items():
            if sha256(p)!=h:raise ValueError('Completed ITC artifact changed')
        return saved
    import torch
    from PIL import Image
    from videoqa_runtime.baseline_selection import BlipItmScorer
    from videoqa_budget32.selection import configure_visual_precision
    torch.set_num_threads(2);configure_visual_precision(torch)
    with exclusive(root/'run.lock'):
        scorer=BlipItmScorer(str(MODEL));params=load_projection_parameters(MODEL/'pytorch_model.bin')
        durable(root/'environment.json',dict(gpu=gpu,pid=os.getpid(),sid=os.getsid(0),torch=torch.__version__,transformers=__import__('transformers').__version__,dtype=str(scorer.model.dtype),utc=utc()))
        # 步骤1：合成仅验官方路径，真实文本不输入答案、选项或人工需求。
        images=[Image.new('RGB',(1280,720),c) for c in ('gray','white','black','gray')]
        validate_once(root,'synthetic',scorer,images,None,None,params)
        for im in images:im.close()
        for q in spec['question_ids']:
            text_once(root,q,spec['inputs'][q]['question'],scorer)
        # 步骤2：固定32图官方ITC分支对齐，不沿用失败投影支持的答案。
        for q,item in spec['validation'].items():
            indices=item['indices'];images=[Image.open(BANK/'bank'/q/f'{i:03d}.png').convert('RGB') for i in indices]
            old=np.load(item['old_feature_file'],allow_pickle=False)[indices]
            text=np.load(root/'text_calls'/(q+'.npy'),allow_pickle=False)
            validate_once(root,q,scorer,images,old,text,params)
            for im in images:im.close()
        # 步骤3：释放GPU模型后做缓存投影/排名；这些CPU成本单列，不称原片E2E。
        del scorer;torch.cuda.empty_cache()
        return analyze(root,spec,params)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True);parser.add_argument('--stage',choices=('prepare','run'),required=True);parser.add_argument('--gpu',default='7')
    args=parser.parse_args()
    if not args.run_id.replace('_','').replace('-','').isalnum():raise ValueError('Invalid run ID')
    root=ROOT/'outputs/analysis'/args.run_id
    try:
        prepare(root) if args.stage=='prepare' else execute(root,args.gpu)
    except BaseException as exc:
        durable(root/('failure_'+str(time.time_ns())+'.json'),dict(error=repr(exc),utc=utc()));raise
