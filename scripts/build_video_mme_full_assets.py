#!/usr/bin/env python3
"""构建 Video-MME 全量测试集资产：题目清单（2700题）与视频资产审计（900个视频）。

因两个阶段依赖不同，需用不同解释器执行（环境均不改动）：
- manifest 子命令：读标注 parquet 生成 data/manifests/video_mme_full_2700.json（schema 与冻结开发50题清单一致），
  并与开发50题清单逐题交叉核验标注一致性；依赖 pandas/pyarrow，
  用 /App/conda/envs/conda_xmake/bin/python 执行。
- audit 子命令：用 PyAV 探测 900 个视频媒体信息并计算 SHA256，生成
  data/manifests/video_mme_full_asset_audit.json（运行时逐视频哈希预检与时长调度排序的数据源），
  随后将清单 video_validation_status 置为 verified 并打印清单最终 SHA256（供运行器数据集注册表冻结）；
  依赖 av，用 /App/conda/envs/explore_video_qa/bin/python 执行。
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARQUET = ROOT / 'data/archives/video_mme_full/videomme/test-00000-of-00001.parquet'
MANIFEST_OUT = ROOT / 'data/manifests/video_mme_full_2700.json'
AUDIT_OUT = ROOT / 'data/manifests/video_mme_full_asset_audit.json'
DEV_MANIFEST = ROOT / 'data/manifests/video_mme_development.json'


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=1) + '\n')
    tmp.replace(path)


def build_manifest():
    """步骤1 读标注 parquet；步骤2 校验字段完备性；步骤3 构造清单行；
    步骤4 与冻结开发50题清单逐题交叉核验；步骤5 写清单文件。"""
    import pandas as pd
    # 步骤1：读取全量标注（2700 行、900 视频、每视频 3 题）
    df = pd.read_parquet(PARQUET)
    assert len(df) == 2700, f'期望2700行，实际{len(df)}'
    assert df['videoID'].nunique() == 900, '期望900个唯一视频'
    assert df.groupby('videoID').size().eq(3).all(), '期望每视频恰好3题'
    assert df['question_id'].is_unique, 'question_id 必须唯一'
    assert df['duration'].isin(['short', 'medium', 'long']).all(), 'duration 分层异常'
    assert df['answer'].isin(list('ABCD')).all(), '答案必须为 A-D'

    # 步骤2：构造与开发清单同 schema 的行；original 保留 parquet 原始字段
    rows = []
    for record in df.to_dict('records'):
        video_id = record['videoID']
        relative = f'videos/video_mme/{video_id}.mp4'
        assert (ROOT / 'data' / relative).exists(), f'视频缺失: {relative}'
        rows.append(dict(
            answer_index='ABCD'.index(record['answer']),
            domain=record['domain'],
            options=list(record['options']),
            original=dict(answer=record['answer'], domain=record['domain'], duration=record['duration'],
                          options=list(record['options']), question=record['question'],
                          question_id=str(record['question_id']), sub_category=record['sub_category'],
                          task_type=record['task_type'], url=record['url'], videoID=record['videoID'],
                          video_id=str(record['video_id'])),
            question=record['question'],
            question_id=str(record['question_id']),
            source_group=f'youtube:{video_id}',
            source_url=record['url'],
            stratum=record['duration'],
            task_type=record['task_type'],
            video_id=video_id,
            video_relative_path=relative,
        ))

    # 步骤3：与冻结开发50题清单交叉核验——证明本 parquet 与开发集标注来源内容一致
    dev_rows = {r['question_id']: r for r in json.loads(DEV_MANIFEST.read_text())['rows']}
    full_rows = {r['question_id']: r for r in rows}
    missing = set(dev_rows) - set(full_rows)
    assert not missing, f'开发清单题目在全量标注中缺失: {sorted(missing)}'
    mismatched = []
    for qid, dev in dev_rows.items():
        full = full_rows[qid]
        for field in ('question', 'options', 'answer_index', 'video_id', 'stratum', 'task_type', 'domain'):
            if dev[field] != full[field]:
                mismatched.append((qid, field, dev[field], full[field]))
    assert not mismatched, f'与开发清单标注不一致: {mismatched}'

    # 步骤4：写清单；seed 置空表示未抽样（全量 parquet 顺序）
    manifest = dict(
        dataset='video_mme',
        metadata_sha256=sha256(PARQUET),
        rows=rows,
        sampling='no sampling; all 2700 public test questions in parquet order',
        schema_version=1,
        seed=None,
        source=dict(files=['videomme/test-00000-of-00001.parquet'],
                    repo='AI-ModelScope/Video-MME (mirror of lmms-lab/Video-MME)',
                    revision='master', split='test'),
        split='full_test',
        video_validation_status='pending',
    )
    write_json(MANIFEST_OUT, manifest)
    print(f'manifest written: {MANIFEST_OUT} ({len(rows)} rows)')
    print(f'dev-50 cross-check: all {len(dev_rows)} questions match annotation fields')


def build_audit():
    """步骤1 读清单；步骤2 PyAV 探测媒体信息；步骤3 计算 SHA256；
    步骤4 写审计文件；步骤5 清单状态置 verified 并打印最终 SHA256。"""
    import av
    manifest = json.loads(MANIFEST_OUT.read_text())
    assert manifest['video_validation_status'] in ('pending', 'verified'), '清单状态异常'
    videos = {}
    for row in manifest['rows']:
        videos.setdefault(row['video_id'], row['video_relative_path'])

    files, probes = [], []
    for i, (video_id, relative) in enumerate(sorted(videos.items()), 1):
        path = ROOT / 'data' / relative
        # 步骤2：探测时长、分辨率、编码与帧率（时长用于调度器降序排序）
        container = av.open(str(path))
        stream = container.streams.video[0]
        rate = stream.average_rate
        probe = dict(video_id=video_id, duration_seconds=float(stream.duration * stream.time_base),
                     width=stream.codec_context.width, height=stream.codec_context.height,
                     codec=stream.codec_context.name,
                     average_fps=f'{rate.numerator}/{rate.denominator}' if rate else None,
                     declared_frames=int(stream.frames) if stream.frames else None,
                     time_base=str(stream.time_base), start_time=float(stream.start_time or 0))
        container.close()
        probes.append(probe)
        # 步骤3：字节级哈希（运行时预检逐视频复核该值）
        files.append(dict(path=f'data/{relative}', bytes=path.stat().st_size, sha256=sha256(path)))
        if i % 100 == 0:
            print(f'probed+hashed {i}/900', flush=True)

    # 步骤4：审计文件结构与开发50题审计兼容（files[].path/sha256、videos[].duration_seconds 为运行时消费字段）
    audit = dict(generated_at_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
                 dataset_manifest='data/manifests/video_mme_full_2700.json',
                 question_count=len(manifest['rows']), unique_video_count=len(probes),
                 files=files, videos=probes)
    # 时长统计仅作参考；分层以标注 parquet 的官方口径为准（探测时长仅用于调度器降序排序）
    audit['strata'] = dict(short=sum(1 for p in probes if p['duration_seconds'] < 120),
                           medium=sum(1 for p in probes if 120 <= p['duration_seconds'] < 900),
                           long=sum(1 for p in probes if p['duration_seconds'] >= 900))
    # 官方分层与实测时长的边界出入仅作诊断记录，不作为校验条件
    strata_by_video = {r['video_id']: r['stratum'] for r in manifest['rows']}
    edge_cases = [dict(video_id=p['video_id'], labeled=strata_by_video[p['video_id']],
                       measured_seconds=round(p['duration_seconds'], 1))
                  for p in probes
                  if strata_by_video[p['video_id']] != ('short' if p['duration_seconds'] < 120 else 'medium' if p['duration_seconds'] < 900 else 'long')]
    write_json(AUDIT_OUT, audit)
    print(f'audit written: {AUDIT_OUT} ({len(files)} videos, strata_by_duration={audit["strata"]})')
    if edge_cases:
        print(f'note: {len(edge_cases)} videos have labeled stratum differing from measured-duration heuristic '
              f'(annotation is authoritative); first 5: {edge_cases[:5]}')

    # 步骤5：清单置 verified；打印最终 SHA256 供运行器注册表冻结
    manifest['video_validation_status'] = 'verified'
    write_json(MANIFEST_OUT, manifest)
    print(f'manifest final sha256: {sha256(MANIFEST_OUT)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['manifest', 'audit'])
    args = parser.parse_args()
    if args.phase == 'manifest':
        build_manifest()
    else:
        build_audit()
