"""事件定位扩展的三题原图复核；不生成答案，不修改冻结分数或选择。"""
import argparse
import html
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'src'))
from videoqa_runtime.common import read_json, sha256
from videoqa_full.state import rows, durable
from videoqa_audit.region_scope_review import draw_board


def review(root):
    """接口：九题总结中按评分来源选新三题，提供两组Top32及最终32帧，不按对错挑题。"""
    root = Path(root); summary = read_json(root / 'summary.json')
    if summary['status'] != 'completed' or summary['new_qa_calls'] != 0:
        raise ValueError('Scoring population not complete')
    bank = {r['question_id']: r for r in rows()}
    panel = [r['question_id'] for r in summary['records'] if r['scoring_kind'] == 'new']
    if len(panel) != 3: raise ValueError('Independent three-question panel changed')
    page = ['<!doctype html><html lang="zh"><meta charset="utf-8"><title>事件定位独立场景复核</title>',
        '<style>body{font:16px/1.6 sans-serif;max-width:1300px;margin:24px auto}.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}img{max-width:100%}</style>',
        '<h1>事件定位独立场景复核</h1><p>无新问答。先看问题及原RGB，种子排名不等于答案信息；不同查询的绝对概率不作直接比较。</p>']
    records = []; sources = {str(root / 'summary.json'): sha256(root / 'summary.json')}
    # 步骤1：面板只包含新评分题；题型、答案、人工字段均不影响前32或最终选择。
    for q in panel:
        item = read_json(root / 'inputs' / (q + '.json')); pool = read_json(item['pool_path'])
        page += ['<h2>' + q + '</h2><p>' + html.escape(bank[q]['question']) + '</p>',
            '<details><summary>原选项</summary><p>' + html.escape(' / '.join(bank[q]['options'])) + '</p></details>',
            '<p>定位表达：' + html.escape(item['query']['event_query']) + '</p>']
        for kind in ('seeds', 'selected'):
            page += ['<h3>' + ('前32定位种子' if kind == 'seeds' else '最终32粗帧') + '</h3><div class="pair">']
            for condition, label in [('original_locator_coarse_only', 'original'), ('event_locator_coarse_only', 'event')]:
                path = root / 'selections' / condition / (q + '.json'); result = read_json(path)
                if kind == 'seeds':
                    indices = result['plan']['seed_indices']
                    selection = dict(video=pool['video'], selected_frames=sorted(
                        [pool['candidates'][i] for i in indices], key=lambda f: f['source_pts']))
                    path = root / 'review/seeds' / label / (q + '.json')
                    value = dict(question_id=q, selection=selection, scope='Sorted localization seeds, not final QA frames')
                    if path.exists() and read_json(path) != value: raise ValueError('Review source changed')
                    if not path.exists(): durable(path, value)
                sources[str(path)] = sha256(path)
                board = draw_board((str(root), q, kind, label, str(path))); records.append(board)
                url = kind + '/' + q + '/' + label + '.jpg'
                page += ['<div>' + label + '<a href="' + url + '"><img src="' + url + '"></a></div>']
            page += ['</div>']
    # 步骤2：仅保存只读展示身份，语义初审必须另记录，不能从图板自动宣称定位有效。
    page += ['</html>']; (root / 'review/index.html').write_text('\n'.join(page), encoding='utf-8')
    durable(root / 'review/manifest.json', dict(question_ids=panel, boards=len(records),
        sources=sources, records=records, new_model_calls=0, semantic_review='pending'))
    return panel


def main():
    """接口：显式指定已完成分析目录；图板生成不允许产生评分或答案。"""
    parser = argparse.ArgumentParser(); parser.add_argument('--run-id', required=True); args = parser.parse_args()
    if not args.run_id.replace('_', '').isalnum(): parser.error('Invalid run ID')
    review(ROOT / 'outputs/analysis' / args.run_id)


if __name__ == '__main__':
    main()
