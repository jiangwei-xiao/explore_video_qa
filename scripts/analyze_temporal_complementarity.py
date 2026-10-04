"""全量冻结输入的时间覆盖与配对结果描述，不按答案路由、不调用模型。"""
import argparse
from collections import Counter
from fractions import Fraction
import math
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable
from videoqa_audit.query_input_probe import BASE

CURRENT=ROOT/'outputs/full_videoqa/rd12_exp12__qwen32__videomme2700__20261003_r1'


def coverage(selection):
    """接口：只接收视频与帧，不读取答案；固定32时间箱及端点间最大间隔。"""
    duration=float(Fraction(selection['video']['duration_fraction']))
    t=[f['timestamp_seconds']/duration for f in selection['selected_frames']]
    if not t or t!=sorted(t) or any(not 0<=x<1 for x in t):raise ValueError('帧时间异常')
    if len(set(f['source_pts'] for f in selection['selected_frames']))!=len(t):raise ValueError('重复源帧')
    # 步骤1：按真实PTS归一化，不假定帧率恒定；端点空白也计入未覆盖间隔。
    edges=[0.]+t+[1.];gaps=[b-a for a,b in zip(edges,edges[1:])]
    bins=Counter(min(31,int(x*32)) for x in t)
    entropy=-sum((n/len(t))*math.log(n/len(t)) for n in bins.values())/math.log(32)
    # 步骤2：这些仅是时间分布统计，不将覆盖宽解释为证据更充分。
    return dict(frames=len(t),occupied_bins=len(bins),max_gap_fraction=max(gaps),
                span_fraction=t[-1]-t[0],head_fraction=t[0],tail_fraction=1-t[-1],entropy=entropy)


def summarize(records):
    """接口：固定分组内计算计数与时间统计，不优化阈值或生成混合预测。"""
    out=dict(n=len(records),paired=dict(Counter(r['paired'] for r in records)))
    for method in ['uniform','rd']:
        out[method]=dict(correct=sum(r[method+'_correct'] for r in records),
            metrics={k:dict(mean=float(np.mean([r[method][k] for r in records])),
                            median=float(np.median([r[method][k] for r in records])))
                     for k in ['occupied_bins','max_gap_fraction','span_fraction','entropy']}) if records else None
    return out


def main(run):
    """接口：2700题全部配对，时长和固定覆盖箱分组；零生成、不修改输入。"""
    sources={str(Path(__file__)):sha256(__file__)}
    def read(path):sources[str(path)]=sha256(path);return read_json(path)
    rows=read(CURRENT/'per_question.json');records=[]
    if len(rows)!=2700 or len({r['question_id'] for r in rows})!=2700:raise ValueError('题目分母变化')
    # 步骤1：全量读取两种冻结帧，匹配视频身份；答案仅用于事后分组。
    for row in rows:
        q=row['question_id']
        uniform_record=read(BASE/'selections/BASE-Uniform'/(q+'.json'))
        if uniform_record['question_id']!=q or uniform_record['video_id']!=row['video_id']:
            raise ValueError('均匀来源身份不符')
        u=uniform_record['selection']
        rd_record=read(CURRENT/'selections/RD-1.2-exp12'/(q+'.json'))
        if rd_record['question_id']!=q or rd_record['video_id']!=row['video_id']:
            raise ValueError('RD来源身份不符')
        r=rd_record['selection']
        for key in ['path','time_base','start_pts','duration_fraction']:
            if u['video'][key]!=r['video'][key]:raise ValueError('视频身份或时间协议不一致')
        uc=bool(row['methods']['BASE-Uniform']['correct']);rc=bool(row['methods']['RD-1.2-exp12']['correct'])
        records.append(dict(question_id=q,video_id=row['video_id'],stratum=row['stratum'],task_type=row['task_type'],
            development=row['source']=='imported_development',uniform=coverage(u),rd=coverage(r),
            uniform_correct=uc,rd_correct=rc,
            paired='both_correct' if uc and rc else 'uniform_only' if uc else 'rd_only' if rc else 'both_wrong'))
    # 步骤2：在查看结果前固定四档时间箱数量，不按收益选择阈值。
    buckets=[(0,7),(8,15),(16,23),(24,32)]
    result=dict(sources=sources,new_model_calls=0,records=records,overall=summarize(records),
        duration={s:summarize([r for r in records if r['stratum']==s]) for s in ['short','medium','long']},
        paired_groups={p:summarize([r for r in records if r['paired']==p]) for p in ['both_correct','uniform_only','rd_only','both_wrong']},
        coverage_groups={s:{f'{a}-{b}':summarize([r for r in records if (s=='all' or r['stratum']==s) and a<=r['rd']['occupied_bins']<=b])
            for a,b in buckets} for s in ['all','short','medium','long']},
        development=summarize([r for r in records if r['development']]),
        limitation='按已观察结果的描述性分析；覆盖与得失关联不等于因果，不输出可部署路由或联合准确率')
    # 步骤3：结果独立保存，恢复核对指纹，不改写历史选帧或答案。
    path=run/'analysis.json'
    if path.exists():
        if read_json(path)!=result:raise ValueError('恢复输入或结果变化')
    else:run.mkdir(parents=True,exist_ok=True);durable(path,result)
    print(result['overall'])
    for p,v in result['paired_groups'].items():print(p,v)
    for b,v in result['coverage_groups']['all'].items():print(b,v)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True,type=Path)
    main(parser.parse_args().run)
