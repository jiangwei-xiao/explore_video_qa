"""RD-1.2-exp19冻结200题RGB输入；仅CPU解码，不启动Qwen或其他模型。"""
import os
os.environ.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from videoqa_runtime.common import read_json, sha256
from videoqa_full.state import durable, exclusive, utc
from videoqa_audit.query_input_probe import BASE
from videoqa_budget32.frames import FrameProvider

METHOD = 'RD-1.2-exp19'
SOURCE = ROOT/'outputs/analysis/bounded_evidence_supplement200_20261003_r2'
RUN = ROOT/'outputs/methods/rd12_exp19__qwen32__dev200__20261003_r2'
DATA = ROOT/'data/manifests/video_mme_development_200_20260920.json'


def export_one(row):
    """接口：按冻结源帧精确解码、返回逐帧RGB哈希；恢复核验来源，不重新选帧。"""
    q = row['question_id']; p = SOURCE/'questions'/(q+'.json'); record = read_json(p)
    for path, digest in record['sources'].items():
        assert sha256(path) == digest, path
    pool_path = BASE/'pools'/(q+'.json'); pool = read_json(pool_path)
    source = dict(record['sources'])
    for path in [p, SOURCE/'protocol.json', ROOT/'src/videoqa_methods/bounded_evidence_supplement.py', Path(__file__).resolve()]:
        source[str(path)] = sha256(path)
    dest = RUN/'qa_inputs'/(q+'.json')
    # 步骤1：已有完成输入不可覆盖或因得分不理想重选；源身份变化停止。
    if dest.exists():
        item = read_json(dest)
        assert item['sources'] == source and item['method'] == METHOD
        return q, sha256(dest)
    selection = dict(video=pool['video'], selected_frames=[pool['candidates'][i] for i in record['selected_indices']])
    images, hashes = FrameProvider(selection).decode()
    for im in images:
        im.close()
    # 步骤2：候选来源、视频、顺序和预算保持显式；后端再次核验这些RGB而不二次采样。
    durable(dest, dict(question_id=q, video_id=row['video_id'], method=METHOD,
                       selection=selection, rgb_sha256=hashes, sources=source))
    return q, sha256(dest)


def main():
    """接口：四个CPU进程生成正式输入；完成后只写manifest，问答须走既有独立账本入口。"""
    with exclusive(RUN/'prepare.lock'):
        summary = read_json(SOURCE/'summary.json')
        assert summary['status'] == 'completed' and summary['question_count'] == 200
        data = read_json(DATA)['rows']; exports = {}
        # 步骤1：不改变已冻结开发清单，按真实问题ID封装；不读取标准答案用于选择。
        with ProcessPoolExecutor(max_workers=4) as executor:
            pending = [executor.submit(export_one, row) for row in data]
            for future in as_completed(pending):
                q, digest = future.result(); exports[q] = digest
                durable(RUN/'preparation_status.json', dict(status='decoding', completed=len(exports), total=200,
                                                            pid=os.getpid(), sid=os.getsid(0), utc=utc()))
        # 步骤2：标准200题入口复用此清单；当前脚本没有生成后端，不可能隐式消耗问答额度。
        manifest = dict(status='completed', question_count=200, frame_budget=32,
                        dataset_manifest=str(DATA), manifest_sha256=sha256(DATA),
                        parent_run=str(BASE), parent_protocol_sha256=sha256(BASE/'protocol.json'),
                        method=METHOD, question_ids=[r['question_id'] for r in data], exports=exports,
                        configuration='exp07 base with at most four positive conditional-gain/novelty swaps; protected original anchors; minimum coarse-coverage deletion loss; fixed Qwen32 reference protocol',
                        diagnostic=False)
        path = RUN/'qa_inputs/selection_manifest.json'
        if path.exists():
            assert read_json(path) == manifest
        else:
            durable(path, manifest)
        durable(RUN/'preparation_status.json', dict(status='completed', completed=200, new_model_calls=0, utc=utc()))
        print('200 frozen RGB inputs prepared; new model calls 0', flush=True)


if __name__ == '__main__':
    main()
