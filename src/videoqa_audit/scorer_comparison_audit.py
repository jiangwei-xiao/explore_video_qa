"""独立只读评分器审计：源帧精确映射、纯特征身份和完整选择重放。"""
from pathlib import Path
from fractions import Fraction
import math
import numpy as np
from videoqa_runtime.common import ROOT,read_json,sha256
from videoqa_full.state import durable,rows,utc
from videoqa_budget32.selection import INDEX
from videoqa_methods.density_selection import source_time
from videoqa_methods.budget_contract import budget_config,select_exp07
from videoqa_methods.relevance_coverage_budget import select_relevance_coverage
from videoqa_runtime.baseline_selection import select_topk
from videoqa_scorer_comparison.preparation import PROFILES,METHODS,BASE,source_keys


def actual_feature_dim(root):
    """接口：从已通过的实际GPU输出读维度，不把旧768或新模型默认维度写死。"""
    values=[read_json(p) for p in (Path(root)/'workers').glob('coarse_*.json')]
    widths={v['acceptance']['feature_dim'] for v in values if v['acceptance']['status']=='passed'}
    if not values or len(widths)!=1:raise ValueError('Actual accepted feature dimension unavailable')
    return next(iter(widths))


def verify_source_mapping(video,frames,pts):
    """接口：真PTS索引核验第一张不早于目标的源帧，不使用平均FPS近似。"""
    base=Fraction(video['time_base']);origin=video['start_pts']
    # 步骤1：源编号与完整PTS索引对应；有理数目标先向上取整至源时钟。
    for frame in frames:
        i=frame['source_frame_index'];p=frame['source_pts']
        if not 0<=i<len(pts) or int(pts[i])!=p:raise ValueError('Source index/PTS mapping differs')
        target=Fraction(frame.get('requested_fraction',str(frame['requested_seconds'])))
        first=int(np.searchsorted(pts,math.ceil(origin+target/base),side='left'))
        if first!=i:raise ValueError('Not first source frame at or after the exact grid target')
        if abs(float(source_time(frame,video))-frame['timestamp_seconds'])>1e-8:raise ValueError('Relative clock differs')


def audit_coarse(root,require_complete=False):
    """接口：已完成粗池实际数组与原候选/PTS逐项核对；完整模式要求全部2700题。"""
    root=Path(root);dim=actual_feature_dim(root);data=rows();banks={r['question_id']:r for r in data};sources={};byvideo={};n=0;frames=0;maximum=0.
    found={p.stem for p in (root/'checkpoints/coarse').glob('*.json')}
    if not found<=set(banks) or require_complete and found!=set(banks):raise ValueError('Coarse population incomplete or unexpected')
    for q in sorted(found):
        # 步骤1：完成检查点绑定池及特征，不接受从未绑定的现有文件。
        check=read_json(root/'checkpoints/coarse'/(q+'.json'))
        for rel,h in check['files'].items():
            if sha256(root/rel)!=h:raise ValueError('Coarse checkpoint changed')
        pp=root/'coarse/pools'/(q+'.json');fp=root/'coarse/features'/(q+'.npy')
        pool=read_json(pp);f=np.load(fp,allow_pickle=False);old=read_json(BASE/'pools'/(q+'.json'))
        if pool['video']!=old['video'] or pool['candidates']!=old['candidates'][:old['initial_count']]:raise ValueError('Initial pool identity differs')
        if f.shape!=(pool['initial_count'],dim) or f.dtype!=np.float32 or not np.isfinite(f).all() or not np.allclose(np.linalg.norm(f,axis=1),1,atol=1e-5,rtol=0):
            raise ValueError('Actual S2 feature shape/dtype/norm differs')
        scores=pool['initial_scores']
        if len(scores)!=len(f) or not np.isfinite(scores).all() or any(not 0<=s<=1 for s in scores):raise ValueError('ITM probabilities differ')
        vid=banks[q]['video_id'];rec=INDEX/'uniform_records'/(vid+'.json');record=read_json(rec);path=INDEX/record['pts_file']
        if sha256(path)!=record['pts_sha256']:raise ValueError('Complete PTS identity changed')
        verify_source_mapping(pool['video'],pool['candidates'],np.load(path,allow_pickle=False))
        # 步骤2：同视频的真实不同原问题必须仍输出相同纯视觉CLS，排除文本融合误取。
        if vid in byvideo:
            ref=byvideo[vid]
            error=float(np.max(np.abs(ref-f)));maximum=max(maximum,error)
            if error>1e-6:raise ValueError('Actual question dependence found in pure visual features')
        else:byvideo[vid]=f
        sources[str(pp)]=sha256(pp);sources[str(fp)]=sha256(fp);sources[str(rec)]=sha256(rec);sources[str(path)]=record['pts_sha256']
        n+=1;frames+=len(f)
    value=dict(status='passed',scope='complete' if require_complete else 'completed_only',questions=n,
        scored_source_positions=frames,feature_dim=dim,feature_dtype='float32',
        maximum_real_question_cls_difference=maximum,source_mapping='first source PTS at or after exact rational target',
        sources=sources,new_model_calls=0,utc=utc())
    dest=root/'independent_audit'/('coarse_complete.json' if require_complete else 'coarse_interim.json')
    durable(dest,value);return value


def audit_selection(root,profile):
    """接口：固定候选与特征重放三方法，预算/保护/粗参考/无细帧ITM逐项核验。"""
    root=Path(root);dim=actual_feature_dim(root);k=PROFILES[profile];reports={}
    # 步骤1：全量原池重算种子与前缀、等权及加权DP，不接受只检查最终帧数。
    for row in rows():
        q=row['question_id'];pp=root/'pools'/profile/(q+'.json');fp=root/'features'/profile/(q+'.npy')
        pool=read_json(pp);f=np.load(fp,allow_pickle=False);n=pool['initial_count']
        if f.dtype!=np.float32 or f.shape!=(len(pool['candidates']),dim) or pool['new_frames_have_itm_scores'] is not False:
            raise ValueError('Feature rows or fine-frame scoring contract differ')
        record=read_json(INDEX/'uniform_records'/(row['video_id']+'.json'));pts=np.load(INDEX/record['pts_file'],allow_pickle=False)
        verify_source_mapping(pool['video'],pool['candidates'],pts)
        a,t=select_exp07(pool,f,k);b,w=select_relevance_coverage(pool,f,t)
        choices={METHODS[0]:select_topk(pool['candidates'][:n],pool['initial_scores'],k),
            METHODS[1]:[pool['candidates'][i] for i in a],METHODS[2]:[pool['candidates'][i] for i in b]}
        for m,frames in choices.items():
            saved=read_json(root/'selections'/profile/m/(q+'.json'))['selection']['selected_frames']
            if frames!=saved:raise ValueError('Full selection replay differs: '+profile+'/'+m+'/'+q)
        old=read_json(root/'traces'/profile/(q+'.json'))
        if t['region_budgets']!=old['exp07']['region_budgets'] or w['region_budgets']!=old['exp12']['region_budgets']:
            raise ValueError('DP allocation replay differs')
        error=0.
        for actual,saved in zip(t['regions'],old['exp07']['regions']):
            if actual['prefix']!=saved['prefix'] or any(not 0<=i<n for i in actual['reference_coarse_indices']):raise ValueError('Prefix or coarse-only reference differs')
            error=max(error,float(np.max(np.abs(np.asarray(actual['coverage_curve'])-saved['coverage_curve']))))
        if error>1e-6:raise ValueError('Coverage numerical replay differs')
        reports[q]=dict(pool_sha256=sha256(pp),features_sha256=sha256(fp),maximum_abs_error=error,
            frame_counts={m:len(v) for m,v in choices.items()})
    value=dict(status='passed',profile=profile,questions=2700,method_replays=8100,
        source=reports,new_model_calls=0,utc=utc());durable(root/'independent_audit'/(profile+'_selection.json'),value);return value
