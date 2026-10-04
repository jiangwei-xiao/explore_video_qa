"""把已完成的固定案例汇成只读review入口，不生成模型答案或人工评判。"""
from pathlib import Path
import html
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable
from render_rd_failure_panel import render


def main():
    """接口：复用两组固定六题和两题诊断的图板，验证来源，提供可点击原尺寸视图。"""
    base=ROOT/'outputs/analysis'
    run=base/'stage_review_20261004_r1'
    groups=[('exp23_result_cases_20261004_r1','exp23与exp12', ['exp12','exp23']),
            ('wfs_reference_cases6_20261004_r1','发布参考与exp12',['exp12','reference'])]
    labels={'exp12':'RD-1.2-exp12','exp23':'RD-1.2-exp23','reference':'REF-WFS-SB',
            'original':'exp12原输入复测','repair':'人工两帧上下文修复'}
    data={r['question_id']:r for r in read_json(ROOT/'data/manifests/video_mme_full_2700.json')['rows']}
    sources={};manifest=[]
    # 步骤1：只新增两题人工修复的完整输入图板，其余图板沿用已审核缓存。
    diagnostic=ROOT/'outputs/diagnostics/context_repairs_exp12_20261004_r1'
    context=base/'context_repairs_cases2_20261004_r1'
    for q in ['541-1','705-1']:
        for condition in ['original','repair']:
            render((context,q,condition,diagnostic/'results'/condition/(q+'.json')))
    sections=[]
    for folder,title,methods in groups:
        panel=read_json(base/folder/'panel.json')
        sections.append((folder,title,methods,[r['question_id'] for r in panel['cases']]))
    sections.append((context.name,'人工上下文修复诊断',['original','repair'],['541-1','705-1']))
    page=['<!doctype html><html lang="zh-CN"><meta charset="utf-8">',
          '<title>阶段证据review</title><style>body{font:16px/1.6 sans-serif;max-width:1500px;margin:24px auto;padding:0 16px} .pair{display:flex;gap:16px;flex-wrap:wrap}figure{flex:1;min-width:360px;margin:0}img{width:100%}section{border-top:1px solid #bbb;margin-top:32px}summary{cursor:pointer}li{margin-bottom:6px}</style>',
          '<h1>阶段证据review</h1><p>只读入口；所有图板来自冻结源帧，属于原图缩略图而非正式预处理Tensor。方法和案例已被事后分析，不宣称盲审。点击图板查看原尺寸；历史回答默认折叠，不自动判定画面是否充分。</p>',
          '<p>当前最好全量候选仍为exp12的1710/2700；本页人工修复不计入方法成绩。完整结论见仓库阶段复盘文档。</p>']
    # 步骤2：题目与选项原样显示、统一转义；图板逐项核对哈希，不引用不明图片。
    for number,(folder,title,methods,ids) in enumerate(sections,1):
        page.append('<section><h2>'+html.escape(title)+'</h2>')
        for q in ids:
            row=data[q]
            page.append(f'<h3 id="group{number}-{q}">{q}</h3><p>'+html.escape(row['question'])+'</p><ul>')
            page.extend('<li>'+html.escape(o)+'</li>' for o in row['options']);page.append('</ul><div class="pair">')
            answers=[]
            for method in methods:
                path=base/folder/q/(method+'.json');record=read_json(path);image=path.with_suffix('.jpg')
                if sha256(image)!=record['image_sha256'] or sha256(record['source'])!=record['source_sha256']:
                    raise ValueError('Review asset identity mismatch')
                sources.update({str(path):sha256(path),str(image):sha256(image),record['source']:record['source_sha256']})
                url=f'../{folder}/{q}/{method}.jpg'
                page.append('<figure><figcaption>'+html.escape(labels[method])+'</figcaption>'+f'<a href="{url}"><img loading="lazy" src="{url}" alt="{q} {method} 输入图板"></a></figure>')
                answers.append(labels[method]+': '+record['raw_output']+' / '+('正确' if record['correct'] else '错误'))
                manifest.append(dict(question_id=q,method=method,url=url,source=record['source']))
            page.append('</div><details><summary>展开历史回答与标准答案</summary><p>'+html.escape('；'.join(answers))+
                        '</p><p>标准答案：'+'ABCD'[row['answer_index']]+'</p></details>')
            if folder==context.name:
                page.append('<p>这是事前冻结的两帧加入/移除组合，不能单独归因某个组件；新增帧的正式预处理可见性已另行审核。</p>')
        page.append('</section>')
    # 步骤3：不收集或伪造用户判断；所有静态资源位于本地，无外部脚本或网络请求。
    page.append('</html>');run.mkdir(parents=True,exist_ok=True)
    output=run/'index.html';output.write_text('\n'.join(page),encoding='utf-8')
    durable(run/'manifest.json',dict(sources=sources,views=manifest,index_sha256=sha256(output),code_sha256=sha256(__file__),
        new_model_calls=0,reviewer_annotations='none; read-only review entry'))
    print('Review entry ready:',output, 'views',len(manifest),flush=True)


if __name__=='__main__':main()
