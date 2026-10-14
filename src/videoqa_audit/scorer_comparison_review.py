"""评分器对照的有界复核页：固定案例、完整原帧图板和各自原生分数曲线。"""
from pathlib import Path
from fractions import Fraction
import html
from videoqa_runtime.common import ROOT,read_json,sha256
from videoqa_full.state import rows,durable,utc
from videoqa_audit.region_scope_runtime import parent_result
from videoqa_audit.region_scope_review import draw_board
from videoqa_audit.framework_contraction import locations
from videoqa_scorer_comparison.preparation import PROFILES,METHODS

FIXED=('103-1','395-2','383-2','409-3','035-2','544-1')


def freeze_panel(root):
    """接口：六个预列案例优先，其余按冻结题序取改善/退化，最多12题、不择有利样例。"""
    root=Path(root);path=root/'review/panel.json'
    if path.exists():return read_json(path)
    if not (root/'comparison.json').exists():raise ValueError('Complete paired results before choosing supplementary panel')
    selected=list(FIXED);slots=[]
    # 步骤1：问答已冻结后才选分析槽位，题号和正误不进入任何选择算法。
    for profile in PROFILES:
        for method in METHODS[1:]:
            for kind in ('improved','regressed'):
                q=None
                for row in rows():
                    candidate=row['question_id']
                    if candidate in selected:continue
                    a=read_json(parent_result(profile,candidate,method))['correct']
                    b=read_json(root/'qa'/profile/'results'/method/(candidate+'.json'))['correct']
                    match=(not a and b) if kind=='improved' else (a and not b)
                    if match:q=candidate;break
                if q is not None and len(selected)<12:selected.append(q);slots.append(dict(profile=profile,method=method,kind=kind,question_id=q))
    value=dict(question_ids=selected,fixed_ids=list(FIXED),supplemental_slots=slots,
        maximum_questions=12,review_minutes_limit=60,semantic_status='pending_actual_frame_review',
        sources={str(root/'comparison.json'):sha256(root/'comparison.json')},new_model_calls=0)
    durable(path,value);return value


def score_plot(root,profile,q):
    """接口：两个原生概率轴分别展示，仅比较题内排名，不当作跨模型可比的置信度。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root);old=read_json(locations(q,PROFILES[profile])[0]);new=read_json(root/'pools'/profile/(q+'.json'))
    fig,axes=plt.subplots(2,1,figsize=(13,5),sharex=True)
    # 步骤1：各自绘制实际粗分数与原定位窗口，窗口不是已确认语义事件。
    for axis,label,pool in zip(axes,('S1 BLIP1','S2 BLIP2'),(old,new)):
        coarse=pool['candidates'][:pool['initial_count']]
        axis.plot([f['timestamp_seconds'] for f in coarse],pool['initial_scores'],linewidth=.7)
        for r in pool['plan']['regions']:
            axis.axvspan(float(Fraction(r['left_fraction'])),float(Fraction(r['right_fraction'])),alpha=.12,color='orange')
        seeds=pool['plan']['seed_indices'];axis.scatter([coarse[i]['timestamp_seconds'] for i in seeds],[pool['initial_scores'][i] for i in seeds],s=10,color='red')
        axis.set_ylabel(label+' native ITM');axis.set_ylim(0,1)
    axes[-1].set_xlabel('Actual source time in seconds');fig.tight_layout()
    path=root/'review'/profile/q/'scores.png';path.parent.mkdir(parents=True,exist_ok=True);fig.savefig(path,dpi=130);plt.close(fig)
    return dict(path=str(path.relative_to(root/'review')),sha256=sha256(path))


def render(root):
    """接口：复用精确RGB图板，显示两前端三方法与均匀；未初审不输出语义结论。"""
    root=Path(root);panel=freeze_panel(root);bank={r['question_id']:r for r in rows()};records=[];plots=[]
    page=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>BLIP前端全量有界复核</title>',
        '<style>body{font:16px/1.6 sans-serif;max-width:1400px;margin:auto;padding:20px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}img{max-width:100%}details{margin:12px 0}</style>',
        '<h1>BLIP前端全量有界复核</h1><p>最多12题、事后审核。图板为完整原RGB等比缩略，不冒充官方预处理后的可见性。点击可放大；必要文字/动作应核查原帧与模型输入。分数曲线使用各模型原生概率，不作为跨模型相同证据置信度。</p>']
    findings=read_json(root/'review/findings.json') if (root/'review/findings.json').exists() else None
    if findings:
        page+=['<p>初审范围、实际处理器视图和未解决项：<a href="findings.json">完整初审记录</a>。这不是全部图板或原片的逐帧标注。</p>']
    for q in panel['question_ids']:
        row=bank[q];page+=['<h2>'+q+'</h2><p>'+html.escape(row['question'])+'</p>',
            '<p>'+html.escape(' / '.join(row['options']))+'</p>']
        # 步骤1：初审结论和图板分开保存；仅展示实际已审核项，不自动填充语义判断。
        if findings:
            note=next((x for x in findings['cases'] if x['question_id']==q),None)
            if note:
                paragraphs=''.join('<p>'+html.escape(k+': '+str(note[k]))+'</p>'
                    for k in ('profile','qwen32','llava8','observation','cause_status') if k in note)
                page+=['<details><summary>展开实际画面初审</summary>'+paragraphs+'</details>']
        for profile in PROFILES:
            page+=['<h3>'+profile+'</h3>'];plot=score_plot(root,profile,q);plots.append(plot)
            page+=['<a href="'+plot['path']+'"><img loading="lazy" src="'+plot['path']+'"></a>']
            u=parent_result(profile,q,'BASE-Uniform');records.append(draw_board((str(root),q,profile,'uniform',str(u))))
            page+=['<details><summary>均匀对照</summary><img loading="lazy" src="'+profile+'/'+q+'/uniform.jpg"></details>']
            # 步骤2：先展示问题和完整图板，预测/正误放在可展开区域，不宣称盲审。
            for method in METHODS:
                page+=['<h4>'+method+'</h4><div class="pair">']
                for sid,path in [('S1',parent_result(profile,q,method)),('S2',root/'qa'/profile/'results'/method/(q+'.json'))]:
                    label=sid+'_'+method;records.append(draw_board((str(root),q,profile,label,str(path))))
                    item=read_json(path);url=profile+'/'+q+'/'+label+'.jpg'
                    page+=['<div><p>'+sid+'</p><a href="'+url+'"><img loading="lazy" src="'+url+'"></a>',
                        '<details><summary>展开回答</summary><p>'+html.escape(str(item['answer']['parsed_answer']))+' / correct='+str(item['correct'])+'</p></details></div>']
                page+=['</div>']
        page+=['<details><summary>标准答案</summary>'+'ABCD'[row['answer_index']]+'</details>']
    page+=['</html>'];(root/'review/index.html').write_text('\n'.join(page),encoding='utf-8')
    value=dict(question_ids=panel['question_ids'],boards=records,plots=plots,
        new_model_calls=0,status='bounded_actual_frame_review_completed' if findings else 'pending_actual_frame_review',
        findings_sha256=sha256(root/'review/findings.json') if findings else None,
        utc=utc());durable(root/'review/manifest.json',value);return value
