"""局部时间变化诊断：复用粗帧特征，不选择新输入，不运行模型。"""
import os
os.environ.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
import argparse
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable
from videoqa_audit.query_input_probe import BASE


def main(run):
    """接口：保存固定局部范围的邻帧变化，量值只是视觉变化，不解释为语义边界。"""
    panel=ROOT/'configs/analysis/evidence_state_panel_20261003.json'
    sources={str(panel):sha256(panel),str(Path(__file__)):sha256(__file__)}
    cases=[]
    # 步骤1：固定核查范围，使用全部已有粗候选计算相邻变化；不按得分筛选审核帧。
    for q,left,right in [('556-1',38,64),('395-2',265,283),('409-3',72,100)]:
        poolpath=BASE/'pools'/(q+'.json');featurepath=BASE/'features'/(q+'.npy')
        sources[str(poolpath)]=sha256(poolpath);sources[str(featurepath)]=sha256(featurepath)
        pool=read_json(poolpath);n=pool['initial_count'];f=np.load(featurepath,allow_pickle=False)[:n]
        if not np.isfinite(f).all() or len(f)!=n:raise ValueError('Invalid frozen coarse features')
        delta=1-np.clip(np.sum(f[:-1]*f[1:],axis=1),0,1)
        records=[]
        # 步骤2：逐帧保留源身份和两侧差异，不用平均FPS推导源时刻或增加采样。
        for i in range(left,right+1):
            if not 0<i<n-1:raise ValueError('Inspection outside coarse pool')
            records.append(dict(candidate_index=i,source_frame=pool['candidates'][i],
                itm=pool['initial_scores'][i],before_distance=float(delta[i-1]),after_distance=float(delta[i])))
        cases.append(dict(question_id=q,coarse_index_range=[left,right],records=records,
            whole_video_adjacent_distance_median=float(np.median(delta)),
            whole_video_adjacent_distance_p90=float(np.quantile(delta,.9))))
    # 步骤3：完成后可恢复核对；不保存算法预测，也不把较大变化设为正式阈值。
    result=dict(sources=sources,cases=cases,new_model_calls=0,new_selections=0,
        limitation='Visual transitions may be shot cuts or low-information frames; repeated flyer frames can have zero internal change')
    output=run/'transitions.json'
    if output.exists():
        if read_json(output)!=result:raise ValueError('Frozen inputs or analysis changed')
    else:
        run.mkdir(parents=True,exist_ok=True);durable(output,result)
    print(dict(cases=len(cases),frames=sum(len(r['records']) for r in cases),new_model_calls=0),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True)
    main(Path(parser.parse_args().run))
