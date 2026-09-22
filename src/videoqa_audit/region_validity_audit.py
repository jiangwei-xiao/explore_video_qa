"""区域有效性只读盘点：位置是复核线索，不是候选过滤规则。"""
from fractions import Fraction

def describe_regions(record):
    """接口：读取一题冻结选择，输出区域身份、锚点排名、配额和复核位置标记；不使用答案。"""
    s=record['selection'];video=s['video'];c=s['candidates'];scores=s['initial_scores']
    duration=Fraction(video['duration_fraction']);tb=Fraction(video['time_base'])
    order=sorted(range(s['initial_count']),key=lambda i:(-scores[i],c[i]['source_pts'],i));rank={i:k+1 for k,i in enumerate(order)}
    rows=[]
    # 步骤1：以真实PTS和有理数计算锚点位置；最后5%只用于人工审核排队。
    for g in s['plan']['regions']:
        rid=str(g['region_id']);i=g['anchor_index'];t=(c[i]['source_pts']-video['start_pts'])*tb
        selected=s['selection_trace']['region_selected'][rid]
        rows.append(dict(question_id=record['question_id'],region_id=g['region_id'],anchor_index=i,anchor_seconds=float(t),
            anchor_position=float(t/duration),last_five_percent=t*20>=duration*19,anchor_rank=rank[i],anchor_score=scores[i],
            quota=len(selected),selected_indices=selected,seed_indices=g['seed_indices'],weight=g['weight'],refine=g['refine'],
            left=g['left_fraction'],right=g['right_fraction']))
    return rows

def review_candidates(rows,count=8):
    """接口：末端锚点按预算降序、题号和区域号升序选不同题；不根据预测或正确率选题。"""
    selected=[]
    # 步骤2：只排复核顺序，不删除区域或改变任何最终选帧。
    for r in sorted((r for r in rows if r['last_five_percent']),key=lambda r:(-r['quota'],r['question_id'],r['region_id'])):
        if r['question_id'] not in selected:selected.append(r['question_id'])
        if len(selected)==count:break
    return selected
