"""生成全量运行的补跑清单：r1 中因候选帧数 < 16 而无法执行的 6 道题（2 个短视频）。

背景（2026-09-17 修订）：原协议将 16 帧固定预算视为硬性要求，两个短视频
（uF3zNOthLAg≈13.8s→14 个候选、6Z_XNM_iT4g≈10.9s→11 个候选）不足 16 个
去重候选，导致 r1 运行在 133-1 处暂停（5388/5400）。用户裁定：固定预算是
最大选帧上限，视频过短时允许少于 16 帧。

本脚本从全量清单原样抽取 6 行（不改动任何字段），补记来源元数据后写盘并打印
SHA256，供 DATASETS 注册表冻结。清单行必须与父清单逐字节一致（深比较）。
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'data/manifests/video_mme_full_2700.json'
OUT = ROOT / 'data/manifests/video_mme_full_completion_6.json'
QUESTION_IDS = ['133-1', '133-2', '133-3', '134-1', '134-2', '134-3']


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parent = json.loads(PARENT.read_text())
    by_id = {r['question_id']: r for r in parent['rows']}
    rows = []
    for qid in QUESTION_IDS:
        if qid not in by_id:
            sys.exit(f'missing question {qid} in parent manifest')
        rows.append(by_id[qid])
    assert len({r['video_id'] for r in rows}) == 2, 'expected exactly the two short videos'
    manifest = dict(parent)
    manifest.update(dict(
        rows=rows,
        completion_of='videomme_full_uniform_topk_20260916_r1',
        completion_reason='16-frame budget amended to a maximum cap (2026-09-17); these videos yield '
                          '14 (uF3zNOthLAg) and 11 (6Z_XNM_iT4g) distinct 1FPS candidates',
        parent_manifest_sha256=sha256(PARENT),
    ))
    OUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(f'wrote {OUT} with {len(rows)} rows; sha256={sha256(OUT)}')


if __name__ == '__main__':
    main()
