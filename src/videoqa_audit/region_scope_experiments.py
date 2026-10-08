"""exp27全量离线准备：原选择计划固定，补查池按触发核心筛回。"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from collections import Counter
import time
import numpy as np
from videoqa_runtime.common import ROOT,read_json,sha256
from videoqa_full.state import rows,durable,exclusive,digest,utc
from videoqa_audit.framework_contraction import locations,BASE,RUN8,RUN32
from videoqa_methods.budget_contract import select_exp07
from videoqa_methods.region_refinement_scope import restrict_refinement_pool
from videoqa_budget32.frames import validate_frames
from videoqa_audit.coverage_objective_review import frame_identity

METHOD='RD-1.2-exp27'
DEV=ROOT/'data/manifests/video_mme_development_200_20260920.json'
FACTS=ROOT/'outputs/analysis/option_rank_dev200_20261001_r1/witness_evaluation_protocol.json'
FILES=('src/videoqa_methods/region_refinement_scope.py','src/videoqa_audit/region_scope_experiments.py',
       'scripts/run_region_scope_experiments.py','src/videoqa_methods/budget_contract.py',
       'src/videoqa_methods/prefix_coverage_budget.py','src/videoqa_methods/weighted_seed_support.py',
       'src/videoqa_methods/coarse_eligibility.py','src/videoqa_methods/density_selection.py',
       'configs/retrieval_density_v1.json')


def profile_budget(profile):
    """接口：两个已确认工作点分别使用K32与K8，始终只有一个公开预算。"""
    return {'qwen32':32,'llava8':8}[profile]


def offline_one(job):
    """接口：一题一配置筛回合法池、重算前缀/预算，原子保存轻量映射与完整轨迹。"""
    root,q,profile=Path(job[0]),job[1],job[2];k=profile_budget(profile)
    pp,fp,tp,sp=locations(q,k);sources={str(p):sha256(p) for p in (pp,fp,tp,sp)}
    dest=root/'offline'/profile/(q+'.json');trp=root/'traces'/profile/(q+'.json')
    if dest.exists():
        old=read_json(dest)
        if old['sources']!=sources or sha256(trp)!=old['trace_sha256']:raise ValueError('Offline recovery identity differs')
        return old
    # 步骤1：旧单预算审计已绑定历史资产，当前再次核对其来源哈希而不重跑父QA。
    audit=read_json(ROOT/'outputs/analysis/rd12_exp07_contraction_20261008_r1/records'/('K'+str(k))/(q+'.json'))
    if any(audit['sources'][p]!=h for p,h in sources.items()):raise ValueError('Frozen parent source changed')
    start=time.perf_counter();pool=read_json(pp);f32=np.load(fp,allow_pickle=False)
    f=f32 if k==32 else f32[pool['original_candidate_indices']]
    original=read_json(sp)['selection'];old_trace=read_json(tp)
    if k==8:old_trace=old_trace['exp07']
    # 步骤2：上游逐字段保持，只改变细候选资格；粗特征不重新归一化或编码。
    new,nf,scope=restrict_refinement_pool(pool,f,k)
    selected,trace=select_exp07(new,nf,k)
    selection=dict(video=new['video'],selected_frames=[new['candidates'][i] for i in selected]);validate_frames(selection,k)
    unchanged=new['parent_candidate_indices']==list(range(len(pool['candidates'])))
    if unchanged and (selection!=original or trace['region_budgets']!=old_trace['region_budgets']):
        raise ValueError('Unchanged pool failed exact selection control')
    old_keys={frame_identity(x) for x in original['selected_frames']};new_keys={frame_identity(x) for x in selection['selected_frames']}
    n=new['initial_count'];parent_selected={frame_identity(x) for x in original['selected_frames']}
    removed_selected=[i for i in scope['removed_parent_indices'] if frame_identity(pool['candidates'][i]) in parent_selected]
    # 步骤3：仅保存源映射，不复制全部旧池或特征；未来按原行映射恢复即可。
    durable(trp,trace)
    result=dict(question_id=q,profile=profile,method=METHOD,frame_budget=k,selection=selection,
        selected_indices=selected,parent_selected_indices=[new['parent_candidate_indices'][i] for i in selected],
        parent_candidate_indices=new['parent_candidate_indices'],global_feature_indices=new['original_candidate_indices'],
        scope=scope,source_pool=str(pp),source_features=str(fp),sources=sources,
        trace_file=str(trp.relative_to(root)),trace_sha256=sha256(trp),unchanged_pool=unchanged,
        unchanged_selected_sources=old_keys==new_keys,removed_selected_parent_indices=removed_selected,
        selected_fine=sum(i>=n for i in selected),parent_selected_fine=sum(x['candidate_index']>=n for x in original['selected_frames']),
        coarse_reference_count=sum(len(r['reference_coarse_indices']) for r in trace['regions']),
        seed_count=new['plan']['effective_seed_count'],actual_frame_count=len(selected),
        new_model_calls=0,seconds=time.perf_counter()-start)
    durable(dest,result);return result


def offline(root,workers=8):
    """接口：后台全量5400项零模型选择，恢复绑定代码/清单/来源，异常不进入QA。"""
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    data=rows();spec=dict(method=METHOD,profiles=['qwen32','llava8'],question_ids=[r['question_id'] for r in data],
        dataset_manifest_sha256=sha256(ROOT/'data/manifests/video_mme_full_2700.json'),
        development_manifest_sha256=sha256(DEV),facts_sha256=sha256(FACTS),
        code_sha256={n:sha256(ROOT/n) for n in FILES},new_blip_calls=0,new_feature_calls=0,
        max_new_qa_per_profile=200,max_new_qa_total=400)
    with exclusive(root/'offline.lock'):
        protocol=root/'offline_protocol.json'
        if protocol.exists():
            if read_json(protocol)!=spec:raise ValueError('Offline protocol changed')
        else:
            durable(protocol,spec)
            import shutil
            for name in FILES:
                dest=root/'code_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(ROOT/name,dest)
        if (root/'offline_summary.json').exists():
            saved=read_json(root/'offline_summary.json')
            for p,h in saved['files'].items():
                if sha256(p)!=h:raise ValueError('Completed offline output changed')
            return saved
        started=time.perf_counter();records=[]
        # 步骤1：固定两工作点与全量清单，源异常取消待执行CPU任务。
        executor=ProcessPoolExecutor(max_workers=workers)
        try:
            for item in executor.map(offline_one,[(str(root),r['question_id'],p) for p in spec['profiles'] for r in data]):
                records.append(item)
                if len(records)%20==0:
                    import os
                    elapsed=time.perf_counter()-started
                    durable(root/'offline_status.json',dict(completed=len(records),total=5400,pid=os.getpid(),
                        sid=os.getsid(0),elapsed_seconds=elapsed,eta_seconds=elapsed/len(records)*(5400-len(records)),utc=utc()))
        except BaseException:
            executor.shutdown(wait=True,cancel_futures=True);raise
        else:executor.shutdown(wait=True)
        # 步骤2：范围缩小、实际帧数和选择变化分别汇总，不称减少量为无效帧数。
        metrics={}
        for p in spec['profiles']:
            rs=[x for x in records if x['profile']==p]
            if len(rs)!=2700 or len({x['question_id'] for x in rs})!=2700:raise ValueError('Offline population incomplete')
            metrics[p]=dict(questions=2700,parent_fine_count=sum(x['scope']['parent_fine_count'] for x in rs),
                fine_count=sum(x['scope']['fine_count'] for x in rs),
                changed_selected_questions=sum(not x['unchanged_selected_sources'] for x in rs),
                unchanged_pool_questions=sum(x['unchanged_pool'] for x in rs),
                removed_parent_selected_positions=sum(len(x['removed_selected_parent_indices']) for x in rs),
                selected_fine=sum(x['selected_fine'] for x in rs),
                actual_frame_counts=dict(Counter(str(x['actual_frame_count']) for x in rs)))
        files={str(root/'offline'/x['profile']/(x['question_id']+'.json')):sha256(root/'offline'/x['profile']/(x['question_id']+'.json')) for x in records}
        files.update({str(root/x['trace_file']):x['trace_sha256'] for x in records})
        summary=dict(status='passed',records=5400,metrics=metrics,new_model_calls=0,
                     seconds=time.perf_counter()-started,files=files)
        durable(root/'offline_summary.json',summary)
        durable(root/'offline_status.json',dict(status='completed',completed=5400,total=5400,utc=utc()))
        return summary
