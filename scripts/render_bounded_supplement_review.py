"""固定七题差异图板，供离线补充候选审核；无问答或重新评分。"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from PIL import Image, ImageDraw
from videoqa_runtime.common import read_json, sha256
from videoqa_full.state import durable
from videoqa_budget32.frames import FrameProvider
from videoqa_audit.query_input_probe import BASE


def main():
    """接口：两题修复来源、一题目录新增及四道正确控制，不按新答案筛选。"""
    run = ROOT / 'outputs/analysis/bounded_evidence_supplement200_20261003_r2'
    ids = ['409-3', '556-1', '056-2', '103-1', '306-2', '383-2', '872-2']
    durable(run/'review/panel.json', dict(question_ids=ids, new_model_calls=0,
                reason='Two repair cases, sole proxy gain, four previously correct controls'))
    # 步骤1：精确解码每题最多8张差异帧，原32帧及算法输入保持不变。
    for q in ids:
        path = run/'questions'/(q+'.json'); record = read_json(path)
        pool = read_json(BASE/'pools'/(q+'.json'))
        changes = [('add', i) for i in record['trace']['added_indices']] + [('remove', i) for i in record['trace']['removed_indices']]
        ordered = sorted({i for _, i in changes}, key=lambda i: pool['candidates'][i]['source_pts'])
        images, hashes = FrameProvider(dict(video=pool['video'], selected_frames=[pool['candidates'][i] for i in ordered])).decode()
        lookup = dict(zip(ordered, images))
        board = Image.new('RGB', (1280, 210*((len(changes)+3)//4)), 'white'); draw = ImageDraw.Draw(board)
        for j, (kind, i) in enumerate(changes):
            im = lookup[i]; im.thumbnail((316, 178)); x, y = j%4*320, j//4*210
            board.paste(im, (x, y)); draw.text((x, y+180), f'{kind}: c{i} {pool["candidates"][i]["timestamp_seconds"]:.3f}s', fill='black')
        dest = run/'review'/(q+'.jpg'); board.save(dest, quality=94); board.close()
        for im in images:
            im.close()
        # 步骤2：保存来源和RGB哈希；缩略图不当作模型处理后可见性验收。
        durable(dest.with_suffix('.json'), dict(question_id=q, changes=changes, decoded_indices=ordered,
                rgb_sha256=hashes, result_sha256=sha256(path), image_sha256=sha256(dest), new_model_calls=0))
    print('Seven bounded difference boards generated; new model calls 0', flush=True)


if __name__ == '__main__':
    main()
