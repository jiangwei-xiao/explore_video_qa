"""固定开发200题等权输入融合的零调用准备，不附用旧答案。"""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable
from videoqa_audit.query_input_probe import BASE
from videoqa_methods.temporal_mixture import select_equal_temporal_mixture

CURRENT=ROOT/'outputs/full_videoqa/rd12_exp12__qwen32__videomme2700__20261003_r1'


def main(run):
    """接口：固定200题名单，只读取两组冻结源帧，所有题使用同一1/2权重。"""
    manifest=ROOT/'data/manifests/video_mme_development_200_20260920.json'
    rows=read_json(manifest)['rows']
    if len(rows)!=200 or len({r['question_id'] for r in rows})!=200:raise ValueError('冻结名单变化')
    sources={str(p):sha256(p) for p in [manifest,Path(__file__),ROOT/'src/videoqa_methods/temporal_mixture.py']}
    records=[]
    # 步骤1：输入来源逐条绑定，同视频题不能串用另一题的RD选择。
    for row in rows:
        q=row['question_id']
        up=BASE/'selections/BASE-Uniform'/(q+'.json');rp=CURRENT/'selections/RD-1.2-exp12'/(q+'.json')
        u,r=read_json(up),read_json(rp)
        for p in [up,rp]:sources[str(p)]=sha256(p)
        if any(x['question_id']!=q or x['video_id']!=row['video_id'] for x in [u,r]):raise ValueError('问题或视频来源不符')
        selection,trace=select_equal_temporal_mixture(u['selection'],r['selection'])
        result=dict(question_id=q,video_id=row['video_id'],selection=selection,trace=trace,
                    source_hashes={str(up):sources[str(up)],str(rp):sources[str(rp)]},new_model_calls=0)
        # 步骤2：独立产物逐题冻结；恢复只核对，不覆盖修改输入或记录。
        path=run/'selections'/(q+'.json')
        if path.exists():
            if read_json(path)!=result:raise ValueError('恢复选择或来源变化')
        else:durable(path,result)
        records.append(dict(question_id=q,frames=len(selection['selected_frames']),**trace['source_counts'],shared=trace['shared_selected']))
    # 步骤3：只汇总来源分布，不生成任何新方法成绩或混合历史答案。
    summary=dict(sources=sources,questions=200,new_model_calls=0,new_answers=0,records=records,
                 scope='离线固定等权候选；未验证QA收益，不按答案或题型路由')
    path=run/'summary.json'
    if path.exists():
        if read_json(path)!=summary:raise ValueError('汇总恢复变化')
    else:durable(path,summary)
    print('200 selections frozen; model calls 0')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    main(parser.parse_args().run)
