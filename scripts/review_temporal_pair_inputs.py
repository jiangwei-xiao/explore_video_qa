"""固定三题帧对候选的原RGB图板；只解码，不生成答案。"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from videoqa_runtime.common import read_json, sha256
from videoqa_full.state import durable, exclusive
from videoqa_budget32.frames import FrameProvider


def main():
    """接口：核对冻结输入，输出六张只读图板及RGB身份，不借用旧预测。"""
    from PIL import Image, ImageDraw
    probe = ROOT / 'outputs/analysis/temporal_pair_selection200_20261004_r1'
    parent = ROOT / 'outputs/full_videoqa/rd12_exp12__qwen32__videomme2700__20261003_r1'
    out = ROOT / 'outputs/analysis/temporal_pair_review3_20261005_r1'
    with exclusive(out / 'review.lock'):
        # 步骤1：名单按事前指定的关系/物体/文字案例固定，未运行新问答。
        for q in ['383-2', '541-1', '705-1']:
            for label, path in [('exp12', parent / 'selections/RD-1.2-exp12' / (q + '.json')),
                                ('pair', probe / 'selections' / (q + '.json'))]:
                dest = out / f'{q}_{label}.jpg'
                meta = dest.with_suffix('.json')
                if meta.exists():
                    previous = read_json(meta)
                    if previous['source_sha256'] != sha256(path) or previous['image_sha256'] != sha256(dest):
                        raise ValueError('Review source changed')
                    continue
                item = read_json(path)
                images, hashes = FrameProvider(item['selection']).decode()
                frames = item['selection']['selected_frames']
                if len(images) != 32 or len(frames) != 32:
                    raise ValueError('Expected frozen 32-frame input')
                # 步骤2：每相邻两格为实际时间patch一对；缩略图不冒充模型Tensor。
                board = Image.new('RGB', (1280, 1680), 'white')
                draw = ImageDraw.Draw(board)
                for i, (im, frame) in enumerate(zip(images, frames)):
                    im.thumbnail((316, 180))
                    x, y = i % 4 * 320, i // 4 * 210
                    board.paste(im, (x, y))
                    draw.text((x, y + 181), f'{i+1} pair{i//2+1}: {frame["timestamp_seconds"]:.3f}s', fill='black')
                    im.close()
                board.save(dest, quality=92)
                board.close()
                # 步骤3：记录完整来源与像素哈希；没有新答案或语义自动标签。
                durable(meta, dict(question_id=q, condition=label, source=str(path),
                    source_sha256=sha256(path), image_sha256=sha256(dest), rgb_sha256=hashes,
                    frames=frames, new_model_calls=0, new_answers=0,
                    view='source RGB thumbnails; not model preprocessing'))
                print(q, label, flush=True)


if __name__ == '__main__':
    main()
