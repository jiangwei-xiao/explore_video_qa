"""可选区域选择器的开发200题M32控制重放，不计算新特征或答案。"""
import os
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
import argparse
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable
from videoqa_audit.query_input_probe import BASE
from videoqa_methods.optional_region_selection import select_optional_regions

PARENT=ROOT/'outputs/full_videoqa/rd12_exp12__qwen32__videomme2700__20261003_r1'


def main(run):
    """接口：从原池重新构造前缀、曲线和DP，逐题核对原exp12真实选择。"""
    manifest=ROOT/'data/manifests/video_mme_development_200_20260920.json'
    sources={str(p):sha256(p) for p in [Path(__file__),manifest,ROOT/'src/videoqa_methods/optional_region_selection.py']}
    records=[]
    # 步骤1：仅使用原M32完整粗细池，证明新实现没有悄悄改变局部次序或覆盖。
    for row in read_json(manifest)['rows']:
        q=row['question_id'];paths=[BASE/'pools'/(q+'.json'),BASE/'features'/(q+'.npy'),
            PARENT/'traces'/(q+'.json'),PARENT/'selections/RD-1.2-exp12'/(q+'.json')]
        for path in paths:sources[str(path)]=sha256(path)
        pool=read_json(paths[0]);features=np.load(paths[1],allow_pickle=False);old=read_json(paths[2]);selection=read_json(paths[3])
        indices,trace=select_optional_regions(pool,features)
        if indices!=selection['selected_indices'] or trace['region_budgets']!=old['region_budgets']:
            raise ValueError('M32控制离散选择未复现: '+q)
        maximum=0.
        for a,b in zip(trace['regions'],old['regions']):
            if a['region_id']!=b['region_id'] or a['prefix']!=b['prefix'] or a['reference_coarse_indices']!=b['reference_coarse_indices']:
                raise ValueError('前缀或粗参考变化: '+q)
            error=float(np.max(np.abs(np.asarray(a['coverage_curve'])-b['coverage_curve'])))
            if error>1e-6:raise ValueError('覆盖误差超限')
            maximum=max(maximum,error)
        records.append(dict(question_id=q,selection_equal=True,maximum_curve_error=maximum))
    # 步骤2：保存重放证据，不将已有答案冒充新实现的重新测试成绩。
    result=dict(sources=sources,records=records,questions=len(records),new_model_calls=0,
        maximum_curve_error=max(r['maximum_curve_error'] for r in records))
    if len(records)!=200:raise ValueError('开发分母变化')
    path=run/'acceptance.json'
    if path.exists():
        if read_json(path)!=result:raise ValueError('恢复来源或结果变化')
    else:run.mkdir(parents=True,exist_ok=True);durable(path,result)
    print('200 M32 controls exact; max curve error',result['maximum_curve_error'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    main(parser.parse_args().run)
