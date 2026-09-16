"""生成静态审核入口和50题逐组逐帧详情；显示人工判断边界，不覆盖旧匿名复核页面。"""
import argparse
import html
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from videoqa_runtime.common import read_json

LABELS={'direct':'直接证据','context':'必要上下文','topic':'仅主题相关','irrelevant':'无关','uncertain':'不确定'}
ROLES={'required':'必要事实','partial':'部分线索','supporting':'上下文','non_target':'非目标事件','uncertain':'不确定'}
METHODS=['uniform','topk','A','B','C','D']
STYLE='body{max-width:1250px;margin:25px auto;padding:0 18px;font:16px/1.65 sans-serif}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:6px;vertical-align:top}img{width:128px;height:128px;object-fit:contain}details{margin:18px 0}a{color:#1559a1}small{color:#666}'


def page(title,body):
    """统一UTF-8页面外壳；题面与人工文字由调用处HTML转义。"""
    return '<!doctype html><meta charset="utf-8"><title>'+html.escape(title)+'</title><style>'+STYLE+'</style>'+body


def render(audit):
    """把已经组装的最终审核JSON转为可点击静态页面，不新增语义判断或模型调用。"""
    audit=Path(audit);out=audit/'final_review';dest=out/'pages';rows=[]
    if (out/'acceptance.json').exists():raise RuntimeError('最终审核已封存，请在新版本/副本渲染，禁止覆盖当前交付。')
    dest.mkdir(exist_ok=True)
    # 步骤1：保留冻结题序，展示各组原答案但明确此入口不是盲审。
    for q in read_json(audit/'questions.json'):
        qid=q['question_id'];r=read_json(out/'questions'/f'{qid}.json');e=html.escape
        body=f'<a href="../index.html">返回总表</a><h1>{qid}</h1><p>{e(r["question"])}</p><p>'+ '<br>'.join(e(x) for x in r['options'])+'</p>'
        body+='<p>事后AI初审，不是人工金标准。下列“充分性”为人工判断；可见、相关、重复与足够回答分别记录。旧初审帧标签保留，事实修订以r6为准。</p>'
        body+=f'<p>任务：{e(r["operation"])}；证据布局：{e(r["evidence_layout"])}。需要：{e(r["required_evidence"])}</p>'
        body+='<details><summary>384可见性与未确定事项</summary>'
        observations=set();uncertainties=set()
        for p in r['visibility']['pages'].values():observations.add(p['observation']);uncertainties.update(p.get('uncertainties',[]))
        body+=''.join('<p>'+e(t)+'</p>' for t in sorted(observations))+'<p>不确定：'+e('；'.join(sorted(uncertainties)))+'</p></details>'
        body+='<p><a href="../../review/'+qid+'.html">原匿名复核/用户留言页面</a> · <a href="../../full_review/semantic_changes_r6/'+qid+'.json">机制与换帧记录</a> · <a href="../source_windows.html#'+qid+'">已核查原片时间窗口</a></p>'
        for m in METHODS:
            g=r['groups'][m];c=g['counts'];body+=f'<details id="{m}"><summary>{m}：预测{e(str(g["prediction"]))}，标准{e(str(g["reference"]))}，'+('对' if g['correct'] else '错')+'</summary>'
            body+='<p>'+e('；'.join(f'{LABELS[l]}{c[l]}' for l in LABELS))+'，合计16。</p>'
            body+=f'<p>整体：{e(g["sufficiency"])}；{e(g["sufficiency_reason"])}</p>'
            if 'error_localization' in g:body+='<p>错误定位（非因果证明）：'+e(g['error_localization']['reason'])+'</p>'
            body+='<table><tr><th>已列事实/角色</th><th>本组见证数量</th><th>解释限制</th></tr>'
            for fact in r['facts']:body+=f'<tr><td>{e(fact["description"])}（{ROLES[fact["role"]]}）</td><td>{g["facts"][fact["id"]]["count"]}</td><td>{e(fact["limitation"])}</td></tr>'
            body+='</table><table><tr><th>槽位/时间</th><th>384输入</th><th>角色和理由</th><th>重复/新增信息（初审维度）</th></tr>'
            # 步骤2：逐帧图片直接链接原资产；缩略图仅用于导航，点开查看完整384及源帧。
            for f in g['frames']:
                t=f['frame']['timestamp_seconds'];a=f['annotation'];d=f['redundancy'];u=f['input_image']
                body+=f'<tr><td>{f["slot"]}<br>{t:.3f}秒<br>PTS {f["frame"]["source_pts"]}</td><td><a href="{u}"><img loading="lazy" src="{u}" alt="槽位{f["slot"]}的384输入"></a><br><a href="{f["source_image"]}">源帧</a></td><td>{LABELS[a["label"]]}：{e(a["reason"])}</td><td>重复：{e(str(d.get("duplicate")))}；新增：{e(str(d.get("new_information")))}<br>{e(d.get("reason",""))}</td></tr>'
            body+='</table></details>'
        (dest/f'{qid}.html').write_text(page(qid,body),encoding='utf-8')
        cells=' / '.join('✓' if r['groups'][m]['correct'] else '✗' for m in METHODS)
        rows.append(f'<tr><td><a href="pages/{qid}.html">{qid}</a></td><td>{e(q["question"])}</td><td>{e(r["operation"])}</td><td>{cells}</td></tr>')
    # 步骤3：总入口把完整报告、统计、诊断和旧复核并列，避免旧阶段状态冒充最终状态。
    body='<h1>Video-MME开发50题：证据审计</h1><p>50题/300组/4800位置；原帧与384全量AI初审。人工定向诊断8题16次，4改善0退化；不是新算法成绩。无新增优化版或Git提交。</p><p>此页为事后分析，显示历史对错；用户正式逐帧复核未完成，不报告一致率。</p>'
    body+='<p><a href="report.md">最终报告</a> · <a href="next_experiment.md">待确认单因素方案</a> · <a href="category_report.md">类别统计</a> · <a href="errors.csv">128条逐组错误定位</a> · <a href="groups.csv">300组标签数量</a> · <a href="source_windows.html">已核查原片窗口</a> · <a href="../full_review/witnesses_r6/report.md">事实见证表</a> · <a href="../full_review/semantic_changes_r6/report.md">全量机制变化</a> · <a href="../index.html">原匿名复核入口</a></p><p>对错列顺序：均匀 / Top-K / A / B / C / D。</p><table><tr><th>题号</th><th>原问题</th><th>任务操作</th><th>历史对错</th></tr>'+''.join(rows)+'</table>'
    (out/'index.html').write_text(page('开发50题证据审计',body),encoding='utf-8');print('rendered 50 pages, 300 groups, 4800 frame positions')
    # 步骤4：单列实际采样核查窗口，不能把窗口图板称为全片观看或完整视频回放。
    body='<a href="index.html">返回总表</a><h1>已核查原片时间窗口</h1><p>以下为有记录的视觉采样窗口，不含音频/外部字幕，不代表全片穷尽。点图可看源分辨率，PTS/源帧号保留在清单。</p>'
    for q in read_json(audit/'questions.json'):
        qid=q['question_id'];body+=f'<h2 id="{qid}">{qid}</h2>'
        manifests=sorted((audit/'context'/qid).glob('*/frames.json'))
        if not manifests:body+='<p>本题没有额外窗口资产；核查范围为六组已选联合原帧及384视图，未声称原片不存在其他证据。</p>'
        for manifest in manifests:
            r=read_json(manifest);s=r['specification'];root='../'+str(manifest.parent.relative_to(audit))
            body+=f'<details><summary>{s["left"]}—{s["right"]}秒，{s["fps"]} FPS；{html.escape(s.get("reason",""))}</summary><p><a href="{root}/frames.json">精确PTS及源帧清单</a></p>'
            for f in r['frames']:
                uri=f'{root}/{f["source_pts"]}.png';body+=f'<a href="{uri}"><img loading="lazy" src="{uri}" alt="{f["timestamp_seconds"]:.3f}秒 PTS {f["source_pts"]}"></a>'
            body+='</details>'
    (out/'source_windows.html').write_text(page('已核查原片窗口',body),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',required=True);a=p.parse_args();render(a.audit)
