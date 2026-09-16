"""关联冻结50题类型统计与已查看的103-1分段说明，不新增模型调用或修改旧标签。"""
import csv
import html
import json
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from videoqa_runtime.common import ROOT,read_json,write_json,sha256

LAYOUTS={'single_frame':'单帧可提供主要证据','local_sequence':'局部连续过程','distributed_events':'跨时间分散',
         'single_frame_summary':'多事件集中总结','uncertain':'暂不确定'}
NOTES=[
 '男子到桌前，放下塑料袋和纸袋，开始整理外卖物品。末尾接近拿取餐具的准备过程。',
 '5.255秒可见白色叉子；随后手部快速动作、垃圾桶镜头，再转到从纸袋取外卖盒。一个算法片段同时包含目标动作和后续其他动作。',
 '面部特写与手部拆取、准备筷子的画面。13.263秒手部画面获全片最高BLIP分数，但不是叉子清晰展示时刻。',
 '男子摆弄筷子、面盒和面条，尝试使用筷子，镜头在手部、面条、人物之间切换。',
 '继续调整、练习使用筷子并夹面、进食，多个近景和远景反复表现同一大场景；不能把这一长段自动认作新的独立语义事件。',
 '末尾继续用筷子进食，最后画面渐暗；与前段仍属同一进食过程。']


def summarize(out):
    """接口：按原50题标签计算共同成败分布，补充已查看候选的人工内容说明与可点击入口。"""
    out=Path(out);audit=ROOT/'outputs/analysis/evidence_videomme50_20260915_r1/final_review'
    rows=list(csv.DictReader((audit/'questions.csv').open(encoding='utf-8-sig')))
    groups=list(csv.DictReader((audit/'groups.csv').open(encoding='utf-8-sig')))
    e_root=ROOT/'outputs/methods/followups_clocal_query_videomme50_20260915_r1/results/E'
    b_correct={g['question_id']:g['correct']=='True' for g in groups if g['method']=='B'}
    assert all(read_json(e_root/f'{r["question_id"]}.json')['correct']==b_correct[r['question_id']] for r in rows)
    counts={axis:{key:dict(total=len(g:=[r for r in rows if r[axis]==key]),**Counter(r['partition'] for r in g))
                  for key in dict.fromkeys(r[axis] for r in rows)} for axis in ('evidence_layout','operation')}
    assert Counter(r['partition'] for r in rows)=={'all_correct':24,'all_wrong':16,'discordant':10}
    write_json(out/'cohort_statistics.json',dict(counts=counts,questions=rows,cohort_definition='historical six methods; E shares B correctness, partitions unchanged',
        source_sha256={str(audit/'questions.csv'):sha256(audit/'questions.csv'),str(audit/'groups.csv'):sha256(audit/'groups.csv')}))
    # 步骤1：输出分母和共同成败，不把共同失败占比当作单方法错误率。
    lines=['# 50题：方法差异与共同成败复盘','','日期：2026-09-16。统计复用冻结事后AI标注，不使用范围分类器标签替代人工布局。',
           '24共同正确/16共同错误/10方法间变化以历史六组定义；E与B正确题集合相同，纳入E后分区不变。诊断人工修复不是算法组，Q没有最终问答。','',
           '| 方法 | 具体做法 | 正确/50 |','|---|---|---:|',
           '| 均匀 | 1 FPS初始池按候选编号round(linspace)取16帧 | 29 |',
           '| Top-K | 原问题BLIP正匹配概率全局前16，再按时间排序 | 29 |',
           '| A | 分数曲线分段，固定λ=0.5，相关性＋递减覆盖奖励−段内去重，无补查 | 28 |',
           '| B | 同A，λ由原问题范围分类取0.2/0.5/0.8，无补查 | 30 |',
           '| C | B粗选后至多两个±2秒热点补到4 FPS，获得新候选的整段按原配额重选 | 28 |',
           '| D | 使用C扩充池及分数直接全局Top-K，不再做分类/补查位置决策 | 28 |',
           '| E/C-local | C上游相同，固定窗口外粗帧，只重选窗口内原名额 | 30 |',
           '| Q | 原词受限改写后做B式离线评分/选帧；生成失败，不构成问答对照 | 未问答 |','',
           '共同条件：同50题、16唯一帧、同LLaVA/提示词/生成配置；不输入音频、外部字幕。选择器不读选项或标准答案。','',
           '## 按证据布局','', '| 布局 | 总题数 | 六组全对 | 六组全错 | 对错变化 |','|---|---:|---:|---:|---:|']
    for key,v in counts['evidence_layout'].items():lines.append(f'| {LAYOUTS[key]} | {v["total"]} | {v.get("all_correct",0)} | {v.get("all_wrong",0)} | {v.get("discordant",0)} |')
    lines+=['','“单帧”指证据需求，不代表画面仅短暂闪现。局部连续与跨时间分散也不等于分类器LOCAL/GLOBAL。时长持续性尚未作为全量独立轴标注。','',
            '## 按任务操作','', '| 操作 | 总题数 | 六组全对 | 六组全错 | 各方法正确数范围 |','|---|---:|---:|---:|---|']
    for key,v in counts['operation'].items():
        ids={r['question_id'] for r in rows if r['operation']==key}
        correct=[sum(g['question_id'] in ids and g['correct']=='True' for g in groups if g['method']==m) for m in ('uniform','topk','A','B','C','D')]
        lines.append(f'| {key} | {v["total"]} | {v.get("all_correct",0)} | {v.get("all_wrong",0)} | {min(correct)}–{max(correct)}/{v["total"]} |')
    lines+=['','比较和计数最弱；总结/体裁、识别定位相对较好。但小类只有1–4题，不作为总体稳定能力结论。',
            '共同正确不等于选帧充分：264-2均匀画面缺动物、409-3历史输入缺清晰玫瑰盒仍答对，先验/选项线索可能参与；不能据此确认因果。321-2选项重复，不作为干净排序能力证据。','']
    for partition,title in [('all_correct','24道共同正确题'),('all_wrong','16道共同错误题')]:
        lines += [f'## {title}','','| 题号 | 原问题 | 操作 | 证据布局 | 需要的事实/关系 |','|---|---|---|---|---|']
        for r in rows:
            if r['partition']==partition:
                cells=[r['question_id'],r['question'].replace('\n','<br>'),r['operation'],LAYOUTS[r['evidence_layout']],r['required_evidence']]
                lines.append('| '+' | '.join(c.replace('|','／') for c in cells)+' |')
        lines.append('')
    lines+=['## 优化定位的边界','','不能把16道共同错误统一归为选帧问题：103-1/306-2有候选漏选线索；170-2/443-1涉及计数过程、回放或局次；158-3/705-1已有部分文字仍错；736-2等证据或解释仍不明确。',
            '先核查原片所需事实→初始池覆盖→评分排名→分段与配额→最终输入充分性，再决定评分、选择、过程覆盖或问答理解哪一环需要改变。','',
            '## 单题逐环节入口','','[103-1完整展示页](index.html)。先看原片/问题，再展开分类、候选、Top-K、分段、最终帧和预测。候选视频每张显示1秒，非连续动作，不据预览播放时间推断原事件时长。']
    (out/'cohort_report.md').write_text('\n'.join(lines)+'\n')
    # 步骤2：人工分段说明绑定实际看过的全部60候选图板，不声称算法生成语义标签。
    data=read_json(out/'walkthrough.json');assert data['question_id']=='103-1' and len(data['segments'])==6
    descriptions=[];table=['<details><summary>展开AI事后分段内容说明（不是算法事件标签）</summary><table><tr><th>段</th><th>原时间区间</th><th>候选内容说明</th><th>B名额</th></tr>']
    for seg,note in zip(data['segments'],NOTES):
        sid=seg['segment_id'];record=dict(segment_id=sid,display_segment=sid+1,content=note,review_basis='全部60张初始候选384图板已查看；用户尚未复核',
                                        source_start=seg['start_seconds'],source_end=seg['end_seconds'])
        descriptions.append(record)
        table.append(f'<tr><td>{sid+1}</td><td>{seg["start_seconds"]:.3f}–{seg["end_seconds"]:.3f}s</td><td>{html.escape(note)}</td><td>{data["methods"]["B"]["quotas"][sid]}</td></tr>')
    table.append('</table></details>')
    write_json(out/'segment_content_review.json',dict(segments=descriptions,reviewer='AI初审',user_review=None,
        boards={str(p.relative_to(out)):sha256(p) for p in sorted((out/'boards').glob('candidates_*.png'))}))
    # 步骤3：分数曲线以真实源时间作横轴，背景片段和候选编号便于对照。
    width,height=1100,300;duration=data['video']['duration_seconds'];x=lambda t:40+1020*t/duration;y=lambda score:255-210*score
    svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">','<rect width="100%" height="100%" fill="white"/>']
    for seg in data['segments']:
        a=x(seg['start_seconds']);b=x(seg['end_seconds']);color='#edf3fa' if seg['segment_id']%2==0 else '#f7f0e2'
        svg.append(f'<rect x="{a}" y="35" width="{b-a}" height="225" fill="{color}"/><text x="{a+3}" y="28" font-size="12">S{seg["segment_id"]+1}</text>')
    points=' '.join(f'{x(r["timestamp_seconds"]):.2f},{y(r["score"]):.2f}' for r in data['candidates'])
    svg.append(f'<polyline points="{points}" stroke="#1555a0" fill="none" stroke-width="2"/>')
    for r in data['candidates']:
        svg.append(f'<circle cx="{x(r["timestamp_seconds"])}" cy="{y(r["score"])}" r="3" fill="#1555a0"><title>#{r["candidate_index"]} time={r["timestamp_seconds"]} score={r["score"]} rank={r["rank"]}</title></circle>')
    svg.append('<text x="42" y="290" font-size="14">Source time 0 — 60s; BLIP matching score 0 — 1; S1..S6: frozen score-based segments</text></svg>')
    (out/'scores.svg').write_text(''.join(svg))
    page=(out/'index.html').read_text().replace('<div id="segment-notes"></div>','<img src="scores.svg" style="max-width:100%" alt="BLIP分数和分段">'+''.join(table))
    page=page.replace('<h1>103-1：从原片到最终16帧</h1>','<h1>103-1：从原片到最终16帧</h1><p><a href="cohort_report.md">50题共同成败与方法区别报告</a></p>')
    (out/'index.html').write_text(page)
    write_json(out/'artifact_manifest.json',{str(p.relative_to(out)):sha256(p) for p in out.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'})
    print('cohort statistics and six visually reviewed segment descriptions written')


if __name__=='__main__':summarize(sys.argv[1])
