"""exp28离线选择：复用exp27细池，只改变最终区域的合并资格。"""
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import os,time,shutil
import numpy as np
from videoqa_runtime.common import ROOT,read_json,sha256
from videoqa_full.state import rows,durable,exclusive,utc
from videoqa_audit.framework_contraction import locations
from videoqa_audit.region_scope_experiments import DEV,FACTS,profile_budget
from videoqa_methods.region_refinement_scope import restrict_refinement_pool
from videoqa_methods.conditional_window_merge import select_conditional_merge
from videoqa_budget32.frames import validate_frames

METHOD='RD-1.2-exp28'
PARENT_SCOPE=ROOT/'outputs/methods/rd12_exp27_region_scope_20261008_r1'
FILES=('src/videoqa_methods/conditional_window_merge.py','src/videoqa_audit/conditional_merge_experiments.py',
       'scripts/run_conditional_merge_experiments.py','src/videoqa_methods/region_refinement_scope.py',
       'src/videoqa_methods/budget_contract.py','src/videoqa_methods/prefix_coverage_budget.py',
       'src/videoqa_methods/weighted_seed_support.py','src/videoqa_methods/density_selection.py','configs/retrieval_density_v1.json')


def offline_one(job):
    """接口：一题筛回同一exp27池，再按新组织选择；父来源或池变化立即暂停。"""
    root,q,profile=Path(job[0]),job[1],job[2];k=profile_budget(profile)
    pp,fp,tp,sp=locations(q,k);parent_path=PARENT_SCOPE/'offline'/profile/(q+'.json')
    parent=read_json(parent_path);sources={str(p):sha256(p) for p in (pp,fp,tp,sp,parent_path)}
    dest=root/'offline'/profile/(q+'.json');trp=root/'traces'/profile/(q+'.json')
    if dest.exists():
        old=read_json(dest)
        if old['sources']!=sources or sha256(trp)!=old['trace_sha256']:raise ValueError('Offline recovery differs')
        return old
    started=time.perf_counter();pool=read_json(pp);global_features=np.load(fp,allow_pickle=False)
    f=global_features if k==32 else global_features[pool['original_candidate_indices']]
    for p in (pp,fp,tp,sp):
        if parent['sources'][str(p)]!=sources[str(p)]:raise ValueError('Parent input source changed')
    # 步骤1：补查范围/精确目标/粗细特征均与exp27相同，不新增任何候选或前向。
    new,nf,scope=restrict_refinement_pool(pool,f,k)
    if scope!=parent['scope'] or new['parent_candidate_indices']!=parent['parent_candidate_indices']:
        raise ValueError('Scoped pool differs from exp27')
    selected,trace=select_conditional_merge(new,nf,k)
    selection=dict(video=new['video'],selected_frames=[new['candidates'][i] for i in selected]);validate_frames(selection,k)
    old_selection=read_json(sp)['selection'];key=lambda s:[(x['source_frame_index'],x['source_pts']) for x in s['selected_frames']]
    # 步骤2：相同分组和归属必须复现父选择；有结构变化时只记录，不宣称等价。
    before=read_json(PARENT_SCOPE/parent['trace_file'])
    same=len(trace['regions'])==len(before['regions']) and all(
        a['members']==b['members'] and a['seed_indices']==pool['plan']['regions'][i]['seed_indices'] and a['density_weight']==b['density_weight']
        for i,(a,b) in enumerate(zip(trace['regions'],before['regions'])))
    if same and key(selection)!=key(parent['selection']):raise ValueError('Unchanged organization control differs')
    durable(trp,trace)
    value=dict(question_id=q,profile=profile,method=METHOD,frame_budget=k,selection=selection,selected_indices=selected,
        source_pool=str(pp),source_features=str(fp),sources=sources,scope=scope,
        parent_candidate_indices=new['parent_candidate_indices'],global_feature_indices=new['original_candidate_indices'],
        trace_file=str(trp.relative_to(root)),trace_sha256=sha256(trp),actual_frame_count=len(selected),
        original_region_count=trace['original_region_count'],selection_group_count=trace['selection_group_count'],
        added_protected_anchors=trace['selection_group_count']-trace['original_region_count'],
        unchanged_organization=same,unchanged_selected_sources=key(selection)==key(old_selection),
        unchanged_exp27_sources=key(selection)==key(parent['selection']),
        selected_fine=sum(i>=new['initial_count'] for i in selected),new_model_calls=0,seconds=time.perf_counter()-started)
    durable(dest,value);return value


def offline(root,workers=8):
    """接口：固定双配置5400项选择与指纹；恢复不重新评分或构建新特征。"""
    root=Path(root);root.mkdir(parents=True,exist_ok=True);data=rows()
    spec=dict(method=METHOD,profiles=['qwen32','llava8'],question_ids=[r['question_id'] for r in data],
        dataset_manifest_sha256=sha256(ROOT/'data/manifests/video_mme_full_2700.json'),
        development_manifest_sha256=sha256(DEV),facts_sha256=sha256(FACTS),
        parent_protocol_sha256=sha256(PARENT_SCOPE/'offline_protocol.json'),
        code_sha256={n:sha256(ROOT/n) for n in FILES},new_model_calls=0,
        development_gate=dict(parent_correct={'qwen32':136,'llava8':126},stop_if_loss_at_least=4,
                              full_profile='qwen32',full_questions=2700))
    with exclusive(root/'offline.lock'):
        protocol=root/'offline_protocol.json'
        if protocol.exists():
            if read_json(protocol)!=spec:raise ValueError('Frozen offline configuration changed')
        else:
            durable(protocol,spec)
            for n in FILES:
                p=root/'code_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,p)
        if (root/'offline_summary.json').exists():
            value=read_json(root/'offline_summary.json')
            for p,h in value['files'].items():
                if sha256(p)!=h:raise ValueError('Offline export changed')
            return value
        started=time.perf_counter();items=[]
        # 步骤1：同一候选池下逐题独立计算，工作进程不读取答案或人工事实。
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for item in executor.map(offline_one,[(str(root),r['question_id'],p) for p in spec['profiles'] for r in data]):
                items.append(item)
                if len(items)%20==0:durable(root/'offline_status.json',dict(completed=len(items),total=5400,
                    elapsed_seconds=time.perf_counter()-started,pid=os.getpid(),sid=os.getsid(0),utc=utc()))
        # 步骤2：区域数/锚点成本/选帧变化分开汇总，不能将增加区域数当作证据收益。
        metrics={}
        for p in spec['profiles']:
            rs=[x for x in items if x['profile']==p]
            if len(rs)!=2700 or len({x['question_id'] for x in rs})!=2700:raise ValueError('Incomplete offline population')
            metrics[p]=dict(questions=2700,original_groups=sum(x['original_region_count'] for x in rs),
                new_groups=sum(x['selection_group_count'] for x in rs),
                changed_selected_questions=sum(not x['unchanged_selected_sources'] for x in rs),
                changed_vs_exp27=sum(not x['unchanged_exp27_sources'] for x in rs),
                group_count_increase_questions=sum(x['added_protected_anchors']>0 for x in rs),
                all_budget_protected_questions=sum(x['selection_group_count']==x['frame_budget'] for x in rs),
                unchanged_organization=sum(x['unchanged_organization'] for x in rs),selected_fine=sum(x['selected_fine'] for x in rs),
                actual_frame_counts=dict(Counter(str(x['actual_frame_count']) for x in rs)))
        files={str(root/'offline'/x['profile']/(x['question_id']+'.json')):sha256(root/'offline'/x['profile']/(x['question_id']+'.json')) for x in items}
        files.update({str(root/x['trace_file']):x['trace_sha256'] for x in items})
        value=dict(status='passed',records=5400,metrics=metrics,files=files,new_model_calls=0,seconds=time.perf_counter()-started)
        durable(root/'offline_summary.json',value);durable(root/'offline_status.json',dict(status='completed',completed=5400,total=5400,utc=utc()))
        return value
