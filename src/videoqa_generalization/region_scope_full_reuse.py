"""生成前工程修订的显式资产复用；只搬运已验证选择/像素，不修改原运行。"""
from pathlib import Path
import copy
import shutil
import time
from videoqa_runtime.common import ROOT,read_json,sha256
from videoqa_full.state import durable,rows
from . import region_scope_full as study

ENGINEERING={'src/videoqa_generalization/region_scope_full.py',
 'src/videoqa_generalization/region_scope_full_runtime.py','scripts/run_region_scope_full.py'}


def reuse(root,source):
    """接口：已停在GPU前的原运行→新目录缓存；拒绝任何真实/合成调用或科学代码变化。"""
    root,source=Path(root),Path(source)
    if root==source or (root/'offline_protocol.json').exists():raise ValueError('Reuse requires separate unprepared run')
    if list((source/'qa').glob('*/calls/*/*.json')) or list((source/'qa').glob('*/synthetic/*/*.json')):
        raise ValueError('Generated source may not be migrated')
    if read_json(source/'pipeline_status.json')['status']!='paused':raise ValueError('Reuse source must be explicitly stopped before GPU')
    old=read_json(source/'offline_protocol.json');summary=read_json(source/'offline_summary.json')
    if old['task']!='exp27_region_scope_full' or old['method']!=study.METHOD or old['question_ids']!=[r['question_id'] for r in rows()] or summary['records']!=5400:
        raise ValueError('Reuse experiment identity differs')
    for name,h in old['code_sha256'].items():
        if name not in ENGINEERING and sha256(ROOT/name)!=h:raise ValueError('Scientific/inference implementation changed: '+name)
    began=time.perf_counter();root.mkdir(parents=True,exist_ok=True)
    # 步骤1逐文件验证并复制全部已完成离线产物，保持其字节与源计算成本。
    for rel,h in summary['exports'].items():
        p=source/rel
        if not p.resolve().is_relative_to(source.resolve()) or sha256(p)!=h:raise ValueError('Offline reuse artifact changed')
        out=root/rel;out.parent.mkdir(parents=True,exist_ok=True)
        if out.exists():
            if sha256(out)!=h:raise ValueError('Existing reused artifact differs')
        else:shutil.copyfile(p,out)
    profiles={}
    for profile in study.PROFILES:
        run=source/'qa'/profile
        if (run/'prepare_completed.json').exists():
            spec=read_json(run/'protocol.json')
            if spec['method']!=study.METHOD or spec['reference_counts']!=study.EXPECTED[profile]:raise ValueError('Prepared source partition differs')
            for rel,h in spec['input_exports'].items():
                if sha256(run/rel)!=h:raise ValueError('Prepared source input changed')
            profiles[profile]=dict(protocol_sha256=sha256(run/'protocol.json'),prepare_binding_sha256=sha256(run/'prepare_binding.json'),
                prepare_completed_sha256=sha256(run/'prepare_completed.json'),original_cpu_seconds=read_json(run/'prepare_completed.json')['seconds'])
    # 步骤2来源记录独立冻结；新离线入口仅核对已有缓存，不重算选择或解码。
    study.freeze(root/'verified_reuse_source.json',dict(source=str(source),source_offline_protocol_sha256=sha256(source/'offline_protocol.json'),
        source_offline_summary_sha256=sha256(source/'offline_summary.json'),profiles=profiles,source_model_calls=0,
        original_selection_compute_seconds=summary['seconds'],reuse_copy_seconds=time.perf_counter()-began,
        new_implementation_files={n:sha256(ROOT/n) for n in ENGINEERING},reason='Use frozen per-question durations instead of absent historical file-stat fields'))


def import_profile_cache(root,profile,binding_sha):
    """接口：新环境/协议绑定相同科学输入，重建缓存外壳；选择/RGB/像素/提示不变。"""
    root=Path(root);reuse_spec=read_json(root/'verified_reuse_source.json')
    if profile not in reuse_spec['profiles']:return
    source=Path(reuse_spec['source'])
    if list((source/'qa').glob('*/calls/*/*.json')) or list((source/'qa').glob('*/synthetic/*/*.json')):raise ValueError('Reuse source unexpectedly generated')
    old=source/'qa'/profile;run=root/'qa'/profile;expected=reuse_spec['profiles'][profile]
    if sha256(old/'protocol.json')!=expected['protocol_sha256'] or sha256(old/'prepare_binding.json')!=expected['prepare_binding_sha256']:
        raise ValueError('Reuse source binding changed')
    spec=read_json(old/'protocol.json');binding=read_json(run/'prepare_binding.json');oldbinding=read_json(old/'prepare_binding.json')
    for key in ('model_files','video_stats','parent_protocol_sha256','development_protocol_sha256'):
        if binding[key]!=oldbinding[key]:raise ValueError('Reuse hardware-independent input binding differs: '+key)
    # 步骤1真实输入必须保留原模型/任务/环境及未改变的推理代码。
    if spec['environment']!=study.environment(study.PYTHONS[profile]) or spec['framework']!=study.framework_identity():
        raise ValueError('Reuse preprocessing environment differs')
    for name,h in spec['code_sha256'].items():
        if name not in ENGINEERING and sha256(ROOT/name)!=h:raise ValueError('Reuse inference code changed')
    completed=0
    for rel,h in spec['input_exports'].items():
        p=old/rel
        if sha256(p)!=h:raise ValueError('Prepared source cache changed')
        value=read_json(p);study.check_slot(value,profile,p.stem);new=copy.deepcopy(value)
        for source_path,source_hash in value['sources'].items():
            if sha256(source_path)!=source_hash:raise ValueError('Reuse source ancestry changed')
        new['prepare_binding_sha256']=binding_sha
        old_selection=source/'selections'/profile/study.METHOD/(p.stem+'.json');new_selection=root/'selections'/profile/study.METHOD/(p.stem+'.json')
        if sha256(old_selection)!=sha256(new_selection):raise ValueError('Reused selection bytes differ')
        if str(old_selection) in new['sources']:new['sources'][str(new_selection)]=new['sources'].pop(str(old_selection))
        # 步骤2仅改运行绑定及同字节选择的来源位置；实际媒体、像素、Token和引用完全保留。
        if any(new[k]!=value[k] for k in ('items','groups','reference_kind','duration_seconds')):raise ValueError('Scientific input changed during migration')
        study.freeze(run/rel,new);completed+=1
    study.freeze(run/'input_reuse_acceptance.json',dict(status='passed',source=str(old),inputs=completed,
        original_cpu_seconds=expected['original_cpu_seconds'],new_decode_calls=0,new_model_calls=0,
        unchanged_selection_rgb_pixels_tokens_and_references=True))
