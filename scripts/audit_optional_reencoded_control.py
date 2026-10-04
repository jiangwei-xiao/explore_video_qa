"""将新编码的重合特征还原到旧M32池，检查编码数值差异是否改变旧选择。"""
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
from videoqa_budget32.frames import FrameProvider

SOURCE=ROOT/'outputs/analysis/optional_seed64_pool200_20261004_r1'
PARENT=ROOT/'outputs/full_videoqa/rd12_exp12__qwen32__videomme2700__20261003_r1'


def main(run):
    """接口：只做CPU控制重放及必要的RGB核验，不解读新准确率、不新增前向。"""
    if read_json(run/'summary.json')['status']!='passed':raise ValueError('准备尚未完成')
    records=[]
    # 步骤1：通过源身份重排新特征，不能用不同池的候选编号直接对应。
    for row in read_json(SOURCE/'summary.json')['records']:
        q=row['question_id'];old=read_json(BASE/'pools'/(q+'.json'));new=read_json(SOURCE/'pools'/(q+'.json'))
        fresh=np.load(run/'features'/(q+'.npy'),allow_pickle=False,mmap_mode='r')
        lookup={(f['source_pts'],f['source_frame_index']):f['candidate_index'] for f in new['candidates']}
        mapped=fresh[[lookup[(f['source_pts'],f['source_frame_index'])] for f in old['candidates']]]
        selected,trace=select_optional_regions(old,mapped)
        expected=read_json(PARENT/'selections/RD-1.2-exp12'/(q+'.json'))['selected_indices']
        changed=selected!=expected;rgb_equal=None
        # 步骤2：若离散选择不同，另看模型实际RGB是否相同；不将小数误差自动忽略。
        if changed:
            images,hashes=FrameProvider(dict(video=old['video'],selected_frames=[old['candidates'][i] for i in selected])).decode()
            for image in images:image.close()
            parent=read_json(PARENT/'imported/results/RD-1.2-exp12'/(q+'.json'))
            rgb_equal=hashes==parent['rgb_sha256']
        records.append(dict(question_id=q,selection_equal=not changed,rgb_equal_when_changed=rgb_equal,
            new_selected=selected,old_selected=expected,region_budgets=trace['region_budgets']))
    # 步骤3：记录差异而不是修正到有利结果，任何后续缓存政策须整体冻结。
    result=dict(code_sha256=sha256(__file__),source_summary_sha256=sha256(run/'summary.json'),records=records,
        changed_ids=[r['question_id'] for r in records if not r['selection_equal']],
        changed_rgb_ids=[r['question_id'] for r in records if r['rgb_equal_when_changed'] is False],new_model_calls=0)
    path=run/'reencoded_control.json'
    if path.exists() and read_json(path)!=result:raise ValueError('控制恢复变化')
    if not path.exists():durable(path,result)
    print('changed selections',result['changed_ids'],'changed RGB',result['changed_rgb_ids'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True,type=Path)
    main(parser.parse_args().run)
