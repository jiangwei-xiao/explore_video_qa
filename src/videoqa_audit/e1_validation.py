"""E1生成前的只读协议核验；历史答案不进入选择计算。"""
import numpy as np
from videoqa_runtime.common import ROOT,read_json,sha256
from videoqa_runtime.baseline_selection import ProtocolError
from .density_reward_decay import select_density_decay
from .local_distance_scale import select_local_distance

def verify_selection(selection,features,cfg,qid):
    """核对独立扫描与冻结离线池，再重算两遍选择；不一致时须在问答前暂停。"""
    base=ROOT/'outputs/methods/density_extension200_20260920_r1'
    ref=read_json(base/'combined_results/v1'/f'{qid}.json');old=ref['selection']
    # 步骤1：元数据、边界、补查映射必须完全一致；图像资产路径允许属于本次目录。
    for key in ('video','candidates','initial_count','plan','refinement','new_candidate_count'):
        if selection[key]!=old[key]:raise ProtocolError('Upstream mismatch: '+qid+' '+key)
    if not np.allclose(selection['initial_scores'],old['initial_scores'],atol=1e-6,rtol=0):raise ProtocolError('Scores changed')
    manifest=read_json(ROOT/'data/manifests/video_mme_development_200_20260920.json')
    origin=ROOT/'outputs/methods/retrieval_density_v1_videomme50_20260920_r1' if qid in manifest['old_question_ids'] else base
    fp=origin/ref['feature_file']
    if sha256(fp)!=ref['feature_sha256']:raise ProtocolError('Historical feature bytes changed')
    frozen=np.load(fp,allow_pickle=False)
    if features.shape!=frozen.shape or not np.allclose(features,frozen,atol=1e-6,rtol=0):raise ProtocolError('Features changed')
    # 步骤2：配额只从本次新算特征产生，不读离线配额来驱动选择。
    rows=selection['candidates'];plan=selection['plan'];video=selection['video']
    _,budget=select_density_decay(rows,features,plan,video,cfg)
    ids,trace=select_local_distance(rows,features,plan,video,cfg,budget['region_budgets'])
    if budget!=selection['budget_trace'] or trace!=selection['selection_trace'] or ids!=selection['selected_indices']:
        raise ProtocolError('Two-pass selection cannot replay')
    offline=read_json(ROOT/'outputs/analysis/local_distance_scale_200_20260922_r1/records'/f'{qid}.json')
    if ids!=offline['selected_indices'] or trace['region_budgets']!=offline['budgets_D1']:
        raise ProtocolError('Discrete offline selection mismatch')
    if selection['selected_frames']!=[rows[i] for i in ids] or len(ids)!=len(set(ids)) or len(ids)!=16:
        raise ProtocolError('Invalid final inputs')
    return dict(score_max_abs_error=float(np.max(np.abs(np.asarray(selection['initial_scores'])-old['initial_scores']))),feature_max_abs_error=float(np.max(np.abs(features-frozen))))
