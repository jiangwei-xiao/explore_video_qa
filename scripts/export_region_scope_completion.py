"""补齐exp27逐采样目标的完整映射审计，不执行选帧或模型调用。"""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys,argparse
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable,utc,rows
from videoqa_methods.region_refinement_scope import restrict_refinement_pool
from videoqa_audit.region_scope_experiments import profile_budget


def export_one(job):
    """接口：重建原目标映射并逐项比对已测源子集；不重新选择前缀或生成答案。"""
    root,profile,q=Path(job[0]),job[1],job[2]
    parent=root/'offline'/profile/(q+'.json');item=read_json(parent)
    dest=root/'target_mappings'/profile/(q+'.json')
    if dest.exists():
        old=read_json(dest)
        if old['source_record_sha256']!=sha256(parent):raise ValueError('Export source changed')
        return str(dest),sha256(dest)
    # 步骤1：读取原冻结池和真实特征行，仅调用源筛选函数，不调用选择器。
    for p,h in item['sources'].items():
        if sha256(p)!=h:raise ValueError('Frozen source changed')
    pool=read_json(item['source_pool']);f32=np.load(item['source_features'],allow_pickle=False,mmap_mode='r')
    features=f32 if profile=='qwen32' else f32[pool['original_candidate_indices']]
    new,nf,trace=restrict_refinement_pool(pool,features,profile_budget(profile))
    # 步骤2：导出须与已测记录的源子集/全局特征行完全一致，不能补造新候选。
    if new['parent_candidate_indices']!=item['parent_candidate_indices'] or new['original_candidate_indices']!=item['global_feature_indices']:
        raise ValueError('Reconstructed source mapping differs')
    if trace!=item['scope'] or not np.array_equal(nf,features[new['parent_candidate_indices']]):
        raise ValueError('Reconstructed scope or feature rows differ')
    # 步骤3：精确目标、状态和触发依据完整落盘，旧选择/QA文件保持只读。
    durable(dest,dict(question_id=q,profile=profile,source_record=str(parent),source_record_sha256=sha256(parent),
        original_regions=pool['plan']['regions'],triggering_core_windows=trace['windows'],
        refinement=new['refinement'],parent_candidate_indices=new['parent_candidate_indices'],
        global_feature_indices=new['original_candidate_indices'],new_model_calls=0,new_selection_calls=0))
    return str(dest),sha256(dest)


def main():
    """接口：显式运行目录，只补齐完成审计元数据，不加载GPU或修改模型协议。"""
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-id',required=True);args=p.parse_args()
    if not args.run_id.replace('_','').replace('-','').isalnum():p.error('Invalid run-id')
    root=ROOT/'outputs/methods'/args.run_id
    if read_json(root/'independent_completion_audit.json')['status']!='passed':raise ValueError('QA must already be complete')
    data=rows();jobs=[(str(root),profile,r['question_id']) for profile in ('qwen32','llava8') for r in data]
    files={}
    # 步骤1：CPU有界重建；输出数量与固定全量名单闭合。
    with ProcessPoolExecutor(max_workers=8) as executor:
        for path,h in executor.map(export_one,jobs):files[path]=h
    if len(files)!=5400:raise ValueError('Target mapping population incomplete')
    durable(root/'target_mapping_export_audit.json',dict(status='passed',records=5400,files=files,
        new_model_calls=0,new_selection_calls=0,source_code_sha256=sha256(ROOT/'src/videoqa_methods/region_refinement_scope.py'),utc=utc()))
    print('5400 target mappings verified and exported',flush=True)


if __name__=='__main__':main()
