"""exp24第一阶段：冻结八题实际竞争银行，独立视觉提取及门槛判断，最终问答0。"""
import os
os.environ.update(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
import argparse
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
from videoqa_runtime.common import read_json, sha256, offline_environment
from videoqa_full.state import durable, save_array, exclusive, utc, digest
from videoqa_audit.coverage_objective_review import frame_identity, selected_indices
from videoqa_methods.visual_state_features import VisualStateEncoder, encode_visual_states, competition_snapshot
from videoqa_budget32.frames import FrameProvider
from analyze_coverage_objectives import Reader, load_pair, BASE

MODEL = Path("/home/models/Qwen/Qwen2.5-VL-7B-Instruct")
PYTHON = "/App/conda/envs/explore_video_qa_lmms_qwen/bin/python"
QUESTIONS = ("035-2", "018-2", "363-2", "409-3", "556-1", "383-2", "003-2", "006-1")
WITNESSES = {"035-2": [55, 56, 59], "018-2": [5, 6, 12], "363-2": [71, 74, 76],
             "409-3": [77, 80], "556-1": [41, 42, 43], "383-2": [944], "003-2": [], "006-1": []}


def verify(root):
    """接口：逐来源核验字节；固定编码器和调用银行变化时暂停，禁止混用旧特征。"""
    spec = read_json(root / "protocol.json")
    for path, expected in spec['sources'].items():
        if sha256(path) != expected:
            raise ValueError("Frozen source changed: " + path)
    return spec


def prepare(root, reuse_bank=None):
    """接口：固定两个配置的种子、原选帧和见证，最多768源帧；语义标签仅用于评价。"""
    root.mkdir(parents=True, exist_ok=True)
    if (root / "protocol.json").exists():
        return verify(root)
    reader = Reader()
    reuse_bank = Path(reuse_bank).resolve() if reuse_bank else None
    prior = read_json(Path(reuse_bank)/'protocol.json') if reuse_bank else None
    if prior:
        reader.bind(Path(reuse_bank)/'protocol.json')
    code = [Path(__file__), ROOT / "src/videoqa_methods/visual_state_features.py",
        ROOT / "src/videoqa_methods/prefix_coverage_budget.py", ROOT / "src/videoqa_methods/relevance_coverage_budget.py",
        ROOT / "src/videoqa_methods/weighted_seed_support.py", ROOT / "scripts/analyze_coverage_objectives.py",
        ROOT / "src/videoqa_audit/coverage_objective_review.py", ROOT / "src/videoqa_budget32/frames.py",
        ROOT / "src/videoqa_methods/region_context.py", ROOT / "src/videoqa_full/state.py"]
    for path in code:
        reader.bind(path)
    banks, profiles = {}, {}
    # 步骤1：读取旧轨迹控制；M8只映射其合法源帧，不把M32细帧混作M8候选。
    for q in QUESTIONS:
        bank = {}
        profiles[q] = {}
        original = reader.read(BASE / "pools" / (q + ".json"))
        old_features = np.load(reader.bind(BASE / "features" / (q + ".npy")), allow_pickle=False)
        old_lookup = {frame_identity(c): i for i, c in enumerate(original['candidates'])}
        for profile in ("qwen32", "llava8"):
            pool, prefix, parent = load_pair(profile, q, reader)
            chosen = selected_indices(parent)
            ids = set(pool['plan']['seed_indices']) | set(chosen) | set(WITNESSES[q])
            protection = []
            # 已查看942.775秒船图是独立保护见证，不按预测动态选取。
            if q == "383-2":
                protection = [i for i in chosen if 942.7 <= pool['candidates'][i]['timestamp_seconds'] <= 942.9]
                ids.update(protection)
            old = old_features[[old_lookup[frame_identity(c)] for c in pool['candidates']]]
            profiles[q][profile] = dict(pool=pool, prefix=prefix, parent=parent,
                bank_indices=sorted(ids), parent_selected=chosen, protected_witness_indices=protection)
            for i in ids:
                frame = pool['candidates'][i]
                key = frame_identity(frame)
                if key in bank:
                    if not np.array_equal(bank[key]['old_feature'], old[i]):
                        raise ValueError("Shared source old representation differs")
                else:
                    bank[key] = dict(frame=frame, old_feature=old[i])
        ordered = sorted(bank, key=lambda k: (k[1], k[0]))
        frames = [bank[key]['frame'] for key in ordered]
        hashes = []
        # 步骤2：精确PTS解码分成8图批，避免32帧提供器上限和长期保存全视频RGB。
        for start in range(0, len(frames), 8):
            if prior:
                old_bank = prior['banks'][q]
                if old_bank['frames'] != frames or old_bank['video'] != original['video']:
                    raise ValueError('Reusable bank source sequence differs')
                from PIL import Image
                files = [Path(reuse_bank)/'bank'/q/f'{i:03d}.png' for i in range(start,min(start+8,len(frames)))]
                if any(sha256(reader.bind(p)) != prior['sources'][str(p)] for p in files):
                    raise ValueError('Reusable bank bytes changed')
                images = [Image.open(p).convert('RGB') for p in files]
                values = [__import__('hashlib').sha256(np.asarray(im).tobytes()).hexdigest() for im in images]
                if values != old_bank['rgb_sha256'][start:start+8]:
                    raise ValueError('Reusable RGB identity differs')
            else:
                images, values = FrameProvider(dict(video=original['video'], selected_frames=frames[start:start+8])).decode()
            for ordinal, image in enumerate(images, start):
                destination = root / "bank" / q / f"{ordinal:03d}.png"
                destination.parent.mkdir(parents=True, exist_ok=True)
                image.save(destination)
                image.close()
                reader.bind(destination)
            hashes.extend(values)
        feature_path = root / "bank" / (q + "_blip.npy")
        save_array(feature_path, np.stack([bank[key]['old_feature'] for key in ordered]))
        reader.bind(feature_path)
        video_path = original['video']['path']
        reader.bind(video_path)
        profile_path = root / "profiles" / (q + ".json")
        durable(profile_path, profiles[q])
        reader.bind(profile_path)
        banks[q] = dict(frames=frames, rgb_sha256=hashes, video=original['video'],
            old_feature_file=str(feature_path), profiles_file=str(profile_path), count=len(frames))
        print("Frozen bank", q, len(frames), flush=True)
    count = sum(item['count'] for item in banks.values())
    if count > 768:
        raise ValueError("Pilot source budget exceeded")
    # 步骤3：模型/处理器/运行库绑定原始字节，不修改共享模型或正式环境。
    for path in MODEL.iterdir():
        if path.is_file() and path.suffix in (".json", ".safetensors", ".txt"):
            reader.bind(path)
    import inspect
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
    for cls in (Qwen2_5_VisionTransformerPretrainedModel, Qwen2VLImageProcessor):
        reader.bind(inspect.getfile(cls))
    for path in code:
        dest = root / "code_snapshot" / path.name
        import shutil
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    spec = dict(method="RD-1.2-exp24", parent="RD-1.2-exp12", questions=list(QUESTIONS), banks=banks,
        sources=reader.sources, witnesses=WITNESSES, original_hypotheses_reviewed_before_forward=False,
        actual_real_sources=count, real_source_limit=768, synthetic_source_limit=64, final_qa_limit=5400,
        forward_batch_limit=8, selector_text_inputs="none; original ITM weights reused only downstream",
        rule="post-merger FP32 mean/L2; independent single-frame Qwen video; BF16/FA2",
        numerical_tolerance=1e-6, utc=utc())
    durable(root / "protocol.json", spec)
    return spec


def invoke(root, name, images, encoder):
    """接口：调用前落账，完整特征原子保存；未知返回不允许重跑，恢复不重复视觉前向。"""
    path = root / "calls" / (name + ".json")
    archive = root / "calls" / (name + ".npy")
    identity = dict(protocol_sha256=sha256(root / "protocol.json"), labels_review_sha256=sha256(root / "labels_review.json"), name=name,
        rgb_sha256=[__import__('hashlib').sha256(np.asarray(im.convert('RGB')).tobytes()).hexdigest() for im in images])
    if path.exists():
        old = read_json(path)
        if old['identity'] != identity or old['status'] != 'completed' or sha256(archive) != old['feature_sha256']:
            raise ValueError("Uncertain visual forward; repetition forbidden")
        return np.load(archive, allow_pickle=False), old
    started = time.perf_counter()
    durable(path, dict(identity=identity, status="started_may_forward", source_count=len(images), utc=utc()))
    features, observation = encode_visual_states(images, encoder)
    if observation['rgb_sha256'] != identity['rgb_sha256']:
        raise ValueError("Encoded RGB differs")
    saved_hash = save_array(archive, features)
    result = dict(identity=identity, status="completed", source_count=len(images), observation=observation,
        feature_sha256=saved_hash, seconds=time.perf_counter()-started, utc=utc())
    durable(path, result)
    return features, result


def assess(root, spec):
    """接口：同一实际胜者/已选上下文内比较排序；不跨表征比较分数绝对尺度。"""
    records, protective = [], []
    for q, bank in spec['banks'].items():
        old = np.load(bank['old_feature_file'], allow_pickle=False)
        new = np.load(root / "features" / (q + ".npy"), allow_pickle=False)
        mapping = {frame_identity(f): i for i, f in enumerate(bank['frames'])}
        profiles = read_json(bank['profiles_file'])
        for profile, item in profiles.items():
            pool, parent = item['pool'], item['parent']
            lookup_old = {i: old[mapping[frame_identity(pool['candidates'][i])]] for i in item['bank_indices']}
            lookup_new = {i: new[mapping[frame_identity(pool['candidates'][i])]] for i in item['bank_indices']}
            # 步骤1：固定原配额最后一次实际入选；只有真实落选见证才计入改善门槛。
            for witness in WITNESSES[q]:
                region = next(r for r in parent['regions'] if witness in r['members'])
                quota = parent['region_budgets'][str(region['region_id'])]
                prefix = region['prefix'][:quota]
                if witness in prefix or quota <= 1:
                    records.append(dict(question_id=q, profile=profile, witness_index=witness,
                        status="already_selected" if witness in prefix else "anchor_only_no_replaceable_slot"))
                    continue
                before, winner = prefix[:-1], prefix[-1]
                a = competition_snapshot(pool, region, before, witness, winner, lookup_old)
                b = competition_snapshot(pool, region, before, witness, winner, lookup_new)
                records.append(dict(question_id=q, profile=profile, witness_index=witness, status="compared",
                    old=a, new=b, ranking_flip=not a['witness_ahead'] and b['witness_ahead']))
            # 步骤2：942.775已确认船图保留能力风险，固定其原入选时刻的候选银行，不另挑反例。
            for witness in item['protected_witness_indices']:
                region = next(r for r in parent['regions'] if witness in r['members'])
                quota = parent['region_budgets'][str(region['region_id'])]
                pos = region['prefix'].index(witness)
                if pos == 0:
                    protective.append(dict(question_id=q, profile=profile, protected=True, reason="unchanged_anchor"))
                    continue
                before = region['prefix'][:pos]
                competitors = [i for i in item['bank_indices'] if i in region['members'] and i not in before and i != witness]
                comparisons = []
                for winner in competitors:
                    a = competition_snapshot(pool, region, before, witness, winner, lookup_old)
                    b = competition_snapshot(pool, region, before, witness, winner, lookup_new)
                    comparisons.append(dict(winner=winner, old=a, new=b,
                        newly_outranks_protected=a['witness']['local_score'] >= a['actual_winner']['local_score']-1e-6
                            and b['actual_winner']['local_score'] > b['witness']['local_score']+1e-6))
                protective.append(dict(question_id=q, profile=profile, witness_index=witness, comparisons=comparisons,
                    priority_loss_risk=any(c['newly_outranks_protected'] for c in comparisons)))
    gains = sorted({r['question_id'] for r in records if r.get('ranking_flip') and r['question_id'] in ('035-2','018-2','363-2')})
    risk = any(p.get('priority_loss_risk') for p in protective)
    # 步骤3：数值排序只是辅助门槛；原图标签和保护判断须独立复核后才允许开发问答。
    output = dict(method="RD-1.2-exp24", records=records, protected=protective, primary_improvement_questions=gains,
        numeric_gate_passed=len(gains) >= 2 and not risk, protected_priority_risk=risk,
        human_review_required=True, final_qa_calls=0,
        limitation="Frozen bank and original context only, not full-pool selection, semantic sufficiency, or QA accuracy",
        utc=utc())
    durable(root / "pilot_comparison.json", output)
    return output


def run(root, gpu):
    """接口：固定一张空闲GPU后台执行八题银行，无BLIP/文本/问答前向；完成入口直接恢复。"""
    offline_environment()
    spec = verify(root)
    if (root / "completion.json").exists():
        return read_json(root / "completion.json")
    labels = read_json(root / "labels_review.json")
    if not labels.get('approved_before_forward') or labels.get('protocol_sha256') != sha256(root / 'protocol.json'):
        raise ValueError("Actual source labels must be reviewed before any forward")
    import torch
    from PIL import Image
    from videoqa_budget32.selection import configure_visual_precision
    torch.set_num_threads(2)
    configure_visual_precision(torch)
    with exclusive(root / "run.lock"):
        encoder = VisualStateEncoder(MODEL)
        durable(root / "environment.json", dict(torch=torch.__version__, transformers=__import__('transformers').__version__,
            flash_attn=__import__('flash_attn').__version__, physical_gpu=gpu, pid=os.getpid(), sid=os.getsid(0),
            all_visual_parameters_frozen=not any(p.requires_grad for p in encoder.visual.parameters()), utc=utc()))
        # 步骤1：固定合成横/竖/方图及重复图，正常8上限下核验均值分组与末帧内部复制。
        synthetic = [Image.new("RGB", size, color) for size, color in [((1280,720),"gray"),((720,1280),"white"),((384,384),"black"),((1280,720),"gray")]]
        f, observation = invoke(root, "synthetic", synthetic, encoder)
        if not np.allclose(f[0], f[3], atol=1e-6, rtol=0):
            raise ValueError("Identical synthetic source feature differs")
        for image in synthetic:
            image.close()
        # 步骤2：按事前冻结批计划提取，每张真实源帧一次，不通过观察超时重启任务。
        for q, bank in spec['banks'].items():
            features = []
            for start in range(0, bank['count'], 8):
                images = [Image.open(root / "bank" / q / f"{i:03d}.png").convert("RGB") for i in range(start, min(start+8,bank['count']))]
                values, record = invoke(root, q+"__"+str(start), images, encoder)
                if record['observation']['rgb_sha256'] != bank['rgb_sha256'][start:start+8]:
                    raise ValueError("Frozen bank RGB changed")
                features.append(values)
                for image in images:
                    image.close()
            save_array(root / "features" / (q + ".npy"), np.concatenate(features))
            durable(root / "status.json", dict(status="pilot_encoding", question_id=q, utc=utc()))
        result = assess(root, spec)
        calls = [read_json(p) for p in (root / "calls").glob("*.json")]
        real = sum(c['source_count'] for c in calls if c['identity']['name'] != "synthetic")
        if real != spec['actual_real_sources'] or any(c['status'] != 'completed' for c in calls):
            raise ValueError("Visual call accounting mismatch")
        hashes = {str(p): sha256(p) for directory in (root / "features", root / "calls") for p in directory.iterdir() if p.is_file()}
        completion = dict(status="completed_numeric_gate_passed_review_pending" if result['numeric_gate_passed'] else "completed_numeric_gate_not_passed",
            real_visual_sources=real, synthetic_visual_sources=4, final_qa_calls=0,
            primary_improvement_questions=result['primary_improvement_questions'],
            protected_priority_risk=result['protected_priority_risk'], results=hashes, utc=utc())
        durable(root / "completion.json", completion)
        print(completion['status'], completion['primary_improvement_questions'], flush=True)
        return completion


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", choices=("prepare", "pilot"), required=True)
    parser.add_argument("--gpu", default="7")
    parser.add_argument("--reuse-bank-from", help="仅复用已核验RGB，旧表征和视觉返回不复用")
    args = parser.parse_args()
    if not args.run_id.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Invalid run ID")
    root = ROOT / "outputs/diagnostics" / args.run_id
    try:
        if args.stage == "prepare":
            prepare(root, args.reuse_bank_from)
        else:
            run(root, args.gpu)
    except BaseException as exc:
        durable(root / ("failure_"+str(time.time_ns())+".json"), dict(error=repr(exc), utc=utc()))
        raise
