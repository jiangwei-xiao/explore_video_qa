"""绘制exp23全部粗见证得失题的新旧输入；不伪造尚未生成的答案。"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from videoqa_budget32.frames import FrameProvider
from videoqa_runtime.common import read_json, sha256
from videoqa_full.state import durable


def main():
    """接口：固定审计中全部六题，核验RGB后生成缩略图和来源记录，零模型调用。"""
    from PIL import Image, ImageDraw
    # 步骤1：面板来自完整得失清单，不依据新答案挑选有利案例。
    out = ROOT / 'outputs/analysis/exp23_witness_review_20261004_r1'
    audit = read_json(out / 'witness_changes.json')
    questions = sorted({r['question_id'] for r in audit['summary']['gains'] + audit['summary']['losses']})
    paths = {
        'exp12': ROOT / 'outputs/full_videoqa/rd12_exp12__qwen32__videomme2700__20261003_r1/selections/RD-1.2-exp12',
        'exp23': ROOT / 'outputs/methods/rd12_exp23__m64k32__prepare200__20261004_r1/selections',
    }
    for q in questions:
        for label, folder in paths.items():
            source = folder / (q + '.json')
            item = read_json(source)
            dest = out / 'views' / q / (label + '.jpg')
            meta = dest.with_suffix('.json')
            # 步骤2：恢复必须来源和图像哈希一致，完整记录不重新解码。
            if meta.exists():
                previous = read_json(meta)
                if previous['source_sha256'] != sha256(source) or previous['image_sha256'] != sha256(dest):
                    raise ValueError('Panel fingerprint mismatch')
                continue
            images, hashes = FrameProvider(item['selection']).decode()
            if hashes != item['rgb_sha256']:
                raise ValueError('Decoded RGB differs from frozen selection')
            frames = item['selection']['selected_frames']
            board = Image.new('RGB', (1280, 210 * ((len(images) + 3) // 4)), 'white')
            draw = ImageDraw.Draw(board)
            # 步骤3：等比缩略仅供定位，不能代替小文字的实际模型输入可见性核查。
            for i, (im, frame) in enumerate(zip(images, frames)):
                im.thumbnail((316, 180))
                x, y = i % 4 * 320, i // 4 * 210
                board.paste(im, (x, y))
                draw.text((x, y + 181), f"{i+1}: {frame['timestamp_seconds']:.3f}s", fill='black')
                im.close()
            dest.parent.mkdir(parents=True, exist_ok=True)
            board.save(dest, quality=94)
            board.close()
            durable(meta, dict(question_id=q, method=label, source=str(source),
                               source_sha256=sha256(source), image_sha256=sha256(dest),
                               frames=frames, rgb_sha256=hashes, new_model_calls=0,
                               view='source_thumbnail_not_model_tensor'))
            print(q, label, flush=True)
    print('All changed-witness panels ready; no QA generated.', flush=True)


if __name__ == '__main__':
    main()
