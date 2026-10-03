"""问题地图的有界发现面板：只读取历史答案与精确源帧，不运行模型。"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from videoqa_audit.query_input_probe import BASE, FULL
from videoqa_full.state import durable
from videoqa_runtime.common import read_json
from videoqa_budget32.frames import FrameProvider
from render_rd_failure_panel import render

PANEL = ['819-1', '191-3', '409-3', '397-2']


def contexts(directory):
    """接口：仅检查明确列出的粗候选区间；稀疏预览不能证明首次事件或全片证据不存在。"""
    from PIL import Image, ImageDraw
    # 步骤1：范围来自既有定位或首轮图板缺口，不增加采样、不计算任何分数。
    windows = [('819-1', 0, 120, 5), ('191-3', 0, 32, 1),
               ('409-3', 72, 86, 1), ('397-2', 250, 440, 5)]
    for q, left, right, stride in windows:
        pool = read_json(BASE / 'pools' / (q + '.json'))
        trace = read_json(FULL / 'traces' / (q + '.json'))
        coarse = pool['candidates'][:pool['initial_count']]
        scores = pool['initial_scores']
        ranks = {i: j + 1 for j, i in enumerate(sorted(range(len(coarse)),
                 key=lambda i: (-scores[i], coarse[i]['source_pts'], i)))}
        chosen = [f for f in coarse if left <= f['timestamp_seconds'] <= right][::stride]
        # 步骤2：每页不超过24帧，附原问题排名和当前局部前缀/配额，便于定位落选环节。
        for start in range(0, len(chosen), 24):
            dest = Path(directory) / q / f'context_{left}_{right}_step{stride}_{start // 24}.jpg'
            if dest.with_suffix('.json').exists():
                from videoqa_runtime.common import sha256
                meta = read_json(dest.with_suffix('.json'))
                assert meta['image_sha256'] == sha256(dest)
                assert meta['pool_sha256'] == sha256(BASE / 'pools' / (q + '.json'))
                assert meta['trace_sha256'] == sha256(FULL / 'traces' / (q + '.json'))
                continue
            frames = chosen[start:start + 24]
            images, hashes = FrameProvider(dict(video=pool['video'], selected_frames=frames)).decode()
            board = Image.new('RGB', (1280, 225 * ((len(frames) + 3) // 4)), 'white')
            draw = ImageDraw.Draw(board)
            records = []
            for j, (im, f) in enumerate(zip(images, frames)):
                i = f['candidate_index']
                region = next(r for r in trace['regions'] if i in r['members'])
                position = region['prefix'].index(i) + 1 if i in region['prefix'] else None
                quota = trace['region_budgets'][str(region['region_id'])]
                x, y = j % 4 * 320, j // 4 * 225
                im.thumbnail((316, 178)); board.paste(im, (x, y)); im.close()
                draw.text((x, y + 180), f"{f['timestamp_seconds']:.3f}s c{i} BLIP rank{ranks[i]}", fill='black')
                draw.text((x, y + 197), f"region {region['region_id']} prefix {position} / quota {quota}", fill='black')
                records.append(dict(frame=f, rank=ranks[i], region=region['region_id'], prefix_position=position, quota=quota))
            dest.parent.mkdir(parents=True, exist_ok=True)
            board.save(dest, quality=94); board.close()
            # 步骤3：存入只读来源身份和检查范围，明确这是粗帧预览而非新模型输入。
            from videoqa_runtime.common import sha256
            durable(dest.with_suffix('.json'), dict(records=records, rgb_sha256=hashes,
                    pool_sha256=sha256(BASE / 'pools' / (q + '.json')),
                    trace_sha256=sha256(FULL / 'traces' / (q + '.json')),
                    image_sha256=sha256(dest), stride_in_coarse_candidates=stride, new_model_calls=0))
    print('Bounded coarse contexts rendered; new model calls 0', flush=True)


def main(directory):
    """接口：冻结四题发现范围，复用带RGB核验的图板函数；不能据题面预判失败原因。"""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # 步骤1：选择当前共同失败中的两个首次事件题、一个既有定位题和一个状态变化题。
    protocol = dict(question_ids=PANEL, new_model_calls=0,
                    purpose='bounded discovery, not a representative accuracy sample',
                    selection_reason='first-event constraints (819,191), prior localized evidence (409), state transition (397)',
                    interpretation='Question text motivates inspection only; evidence loss remains unverified.')
    path = directory / 'panel.json'
    if path.exists():
        assert read_json(path) == protocol, '面板已冻结，禁止静默换题'
    else:
        durable(path, protocol)
    # 步骤2：仅画三组已完成的真实输入；来源身份和RGB内容由复用函数核验。
    jobs = []
    for q in PANEL:
        jobs.append((directory, q, 'exp07', FULL / 'imported/results/RD-1.2-exp07' / (q + '.json')))
        for label, method in [('uniform', 'BASE-Uniform'), ('topk', 'BASE-BLIP-TopK')]:
            jobs.append((directory, q, label, BASE / 'results' / method / (q + '.json')))
    with ThreadPoolExecutor(max_workers=2) as executor:
        records = list(executor.map(render, jobs))
    # 步骤3：完成清单与图板分开保存；已有图板恢复时不重复解码，更不启动生成。
    durable(directory / 'complete.json', dict(rendered=len(records), new_model_calls=0))
    print(dict(rendered=len(records), new_model_calls=0), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--contexts', action='store_true')
    args = parser.parse_args()
    (contexts if args.contexts else main)(args.output)
