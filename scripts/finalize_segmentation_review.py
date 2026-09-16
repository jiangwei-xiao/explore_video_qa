"""把六题窗口核查产物放入可点击页面，保持旧实验与旧审计记录不变。"""
import html
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from videoqa_runtime.common import read_json,write_json,sha256


def finalize(out,walkthrough):
    """接口：在独立核查目录生成导航，再复制为旧展示页的子入口，沿用现有HTTP服务根目录。"""
    out=Path(out);walkthrough=Path(walkthrough);summary=read_json(out/'summary.json')
    # 步骤1：每段视频、查看范围和图板清楚标明，不把诊断帧当成历史输入。
    page='<!doctype html><meta charset="utf-8"><title>六题事件分段复核</title><style>body{font:16px/1.7 sans-serif;max-width:1200px;margin:25px auto}img{max-width:100%}video{width:768px;max-width:100%}pre{white-space:pre-wrap}</style><h1>103-1及五题：原片窗口与分段复核</h1><p><a href="../index.html">返回103-1逐阶段展示</a> · <a href="report.md">完整判断与方法反馈</a></p><p>无新增问答/评分。以下是额外原片核查图板，不是实验初始候选池；initial #None仅表示与原候选PTS未精确重合，不代表原池缺少相邻证据。全部判断是AI初审，等待用户复核。</p>'
    for case in summary['cases']:
        qid=case['question_id'];page+=f'<h2 id="{qid}">{qid}</h2><p>{html.escape(case["question"])}</p><p>范围分类：{case["scope"]["label"]}；B配额：{case["quotas"]}。</p><p>分段：'+html.escape(str([(s['segment_id']+1,round(s['start_seconds'],3),round(s['end_seconds'],3)) for s in case['segments']]))+'</p>'
        page+='<p>实际补查窗口：'+html.escape(str([(round(h['left_seconds'],3),round(h['right_seconds'],3)) for h in case['hotspots']]))+'</p>'
        for w in case['windows']:
            path=w['path'];mode='全部源帧' if not w['sampling_fps'] else str(w['sampling_fps'])+' FPS核查采样'
            page+=f'<h3>{w["left"]}–{w["right"]}秒，{mode}</h3><video controls muted preload="metadata" src="{path}/source_window.mp4"></video><p>视频保留窗口内源画面时序，静音；图板{w["frames"]}帧。<a href="{path}/frames.json">源PTS及源帧号</a></p>'
            for board in sorted((out/path).glob('board_*.jpg')):
                rel=str(board.relative_to(out));page+=f'<details><summary>查看{board.name}</summary><a href="{rel}"><img loading="lazy" src="{rel}"></a></details>'
    (out/'index.html').write_text(page)
    manifest={str(p.relative_to(out)):sha256(p) for p in out.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'}
    write_json(out/'artifact_manifest.json',manifest)
    # 步骤2：只增加新目录及入口链接，旧帧、旧候选、旧分类和原结果保持不变。
    target=walkthrough/'segmentation_review';shutil.copytree(out,target)
    page=(walkthrough/'index.html').read_text()
    addition='<p><a href="segmentation_review/index.html">新增：103-1动作逐帧核查与另外五题分段复核</a></p>'
    page=page.replace('<h1>103-1：从原片到最终16帧</h1>','<h1>103-1：从原片到最终16帧</h1>'+addition)
    (walkthrough/'index.html').write_text(page)
    write_json(walkthrough/'artifact_manifest.json',{str(p.relative_to(walkthrough)):sha256(p) for p in walkthrough.rglob('*') if p.is_file() and p!=walkthrough/'artifact_manifest.json'})
    print('review linked without modifying experimental results')


if __name__=='__main__':finalize(sys.argv[1],sys.argv[2])
