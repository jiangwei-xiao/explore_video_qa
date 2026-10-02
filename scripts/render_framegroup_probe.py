#!/usr/bin/env python3
"""生成两题能力诊断的静态复核页：全部视角展示，不高亮正确选项或推导准确率。"""
import argparse
import csv
import html
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable


def relative_file(path,run):
    """接口：只生成运行目录内的相对链接，保证复制复核资产后仍可离线打开。"""
    p=Path(path).resolve()
    if not p.is_file():
        raise ValueError(f'Missing review asset:{p}')
    return p.relative_to(run.resolve()).as_posix()


def render(run,option_run):
    """接口：核对已完成结果，导出全部分数与实际帧；不修改冻结评分及输入资产。"""
    assets=read_json(run/'assets_manifest_r3.json')
    embedded=read_json(run/'embedding_summary.json')
    blip=read_json(run/'blip_summary.json')
    options=read_json(option_run/'summary.json')
    views=read_json(run/'embedding_protocol.json')['views']
    # 步骤1：只读核验全部返回记录和源图；不以页面渲染成功代替实验输入验收。
    for summary in [embedded,blip,options]:
        assert summary['status']=='completed'
        for p,h in summary['artifacts'].items():
            assert sha256(p)==h,p
    labels={'additional_clue':'补充线索组','matched_context':'重复定位画面对照','different_context':'其他场景对照','anchor_context':'已拿枪定位组'}
    rows=[]
    for item in embedded['comparisons']:
        q,view,group=item['question_id'],item['query_variant'],item['group']
        r=dict(item)
        r.update(blip_max=blip['group_scores'][q][view][group]['max'],blip_mean=blip['group_scores'][q][view][group]['mean'])
        if view!='question':
            scores=options['scores'][q][view]
            values=[scores[p] for p in assets['cases'][q]['clips'][group]]
            r.update(option_only_max=max(values),option_only_mean=sum(values)/len(values))
        else:
            r.update(option_only_max=None,option_only_mean=None)
        rows.append(r)
    assert len(rows)==25
    keys=list(rows[0])
    with (run/'score_comparison.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    # 步骤2：每个视角完整显示，不按标准答案、分数改善或播放次序选择展示。
    parts=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">',
        '<title>帧组补充线索对照</title><style>body{max-width:1200px;margin:24px auto;padding:0 16px;font:16px/1.6 sans-serif;color:#222}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #ccc;padding:5px;text-align:left}.scroll{overflow:auto}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:8px}img{width:100%;height:auto}figure{margin:0}small{color:#555}section{margin:32px 0}code{overflow-wrap:anywhere}</style>',
        '<h1>帧组补充线索对照</h1><p>两题人工构造能力诊断，最终问答为0；不是自动方法或准确率成绩。原问题视角为主，全部选项视角仅作辅助观察。逆序不自动视为错误证据。</p>',
        '<p>395-2宣传册只是度假相关线索，没有单独证明随后实际行动；409-3两组来自不同时段，不是单帧因果干预。不同模型及不同文本的绝对分数不能直接比较。</p>',
        '<p><a href="score_comparison.csv">下载完整分数CSV</a></p>']
    linked=[]
    for q,case in assets['cases'].items():
        parts += ['<section><h2>'+q+'</h2><p>'+html.escape(case['question'])+'</p><ul>']
        parts += ['<li>'+html.escape(x)+'</li>' for x in case['options']]
        parts += ['</ul><div class="scroll"><table><tr><th>视角</th><th>帧组</th><th>BLIP最大/均值</th><th>仅选项最大/均值</th><th>单图最大余弦</th><th>平均图像嵌入余弦</th><th>原序视频余弦</th><th>逆序视频余弦</th></tr>']
        def number(v):
            if v is None:return '—'
            assert math.isfinite(v)
            return f'{v:.5f}'
        for r in rows:
            if r['question_id']!=q:continue
            cells=[r['query_variant'],labels[r['group']],number(r['blip_max'])+' / '+number(r['blip_mean']),
                number(r['option_only_max'])+' / '+number(r['option_only_mean']),number(r['image_max_cosine']),
                number(r['image_mean_embedding_cosine']),number(r['video_cosine']),number(r['reversed_video_cosine'])]
            parts.append('<tr>'+''.join('<td>'+html.escape(c)+'</td>' for c in cells)+'</tr>')
        parts.append('</table></div>')
        for group,pts_list in case['clips'].items():
            parts.append('<h3>'+labels[group]+'</h3><div class="grid">')
            for pts in pts_list:
                frame=case['frames'][pts]
                assert sha256(frame['file'])==frame['png_sha256']
                raw=relative_file(frame['file'],run)
                model=relative_file(views[q+'__'+pts]['path'],run)
                linked.extend([raw,model])
                parts.append(f'<figure><a href="{html.escape(raw)}"><img loading="lazy" src="{html.escape(raw)}" alt="{q} {frame["source_seconds"]:.3f}秒"></a><figcaption>{frame["source_seconds"]:.3f}秒 · PTS {pts}<br><a href="{html.escape(model)}">辅助嵌入实际输入视图</a></figcaption></figure>')
            parts.append('</div>')
        parts.append('</section>')
    parts.append('<p>显示层不更改原始输入和结果。原始PNG与辅助嵌入实际视图均可点击；BLIP使用相同源图但自身官方384预处理，不与辅助模型宣称等视觉预算。</p></html>')
    (run/'index.html').write_text('\n'.join(parts),encoding='utf-8')
    # 步骤3：保存生成来源及全部本地链接验收；不把静态链接核查冒充浏览器截图验收。
    sources={str(p):sha256(p) for p in [run/'assets_manifest_r3.json',run/'embedding_summary.json',run/'blip_summary.json',option_run/'summary.json',Path(__file__)]}
    durable(run/'review_export.json',dict(status='completed',source_hashes=sources,score_rows=len(rows),
        checked_local_links=len(linked),browser_render_inspected=False,qa_calls=0,
        outputs={str(run/name):sha256(run/name) for name in ['index.html','score_comparison.csv']}))
    print('Review exported:25 score rows,100 checked image links,QA0',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--run-id',default='framegroup_embedding_pilot2_20261002_r1')
    p.add_argument('--option-run',default='option_only_pilot2_20261002_r1')
    args=p.parse_args()
    import re
    assert all(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',v) for v in [args.run_id,args.option_run])
    render(ROOT/'outputs/analysis'/args.run_id,ROOT/'outputs/analysis'/args.option_run)
