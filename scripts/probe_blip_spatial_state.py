"""固定四题的空间状态诊断：最多32真实BLIP图文前向，最终问答0。"""
import os
os.environ.update(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from videoqa_runtime.common import read_json, sha256, offline_environment
from videoqa_full.state import durable, exclusive, rows, utc
from videoqa_audit.blip_spatial_state import spatial_batch, compare_pair
from analyze_coverage_objectives import BASE

MODEL = Path("/home/models/Salesforce/blip-itm-base-coco")
PARENT = ROOT / "outputs/analysis/reference_selection_review_20261006_r1"
CASES = {
    "035-2": dict(coarse=[55, 56, 59, 61, 62], positive=[0, 1], negative=[2, 4],
                  evidence="机构列表增加下一行；负对照列表不变但其他空框/手状态不同"),
    "018-2": dict(wfs=[0, 1, 2], positive=[0, 2], negative=[0, 1],
                  evidence="CM7→B7；负对照均为CM7"),
    "409-3": dict(coarse=[77, 79, 80], wfs=[4], positive=[0, 3], negative=[1, 2],
                  evidence="持盒→盒打开取枪；负对照79.279/79.346秒同一持盒状态（78秒已切男孩，预检时剔除错误标记）"),
    "363-2": dict(coarse=[71, 72, 74], positive=[0, 2], negative=[0, 1],
                  evidence="玩车→枪出现；负对照玩车状态"),
}


def prepare(root):
    """接口：先冻结问题、精确源帧、RGB、已知评价对与32/8调用上限，不读取新答案。"""
    from videoqa_budget32.frames import FrameProvider
    root.mkdir(parents=True, exist_ok=True)
    if (root / "protocol.json").exists():
        return verify(root)
    data = {r["question_id"]: r for r in rows()}
    sources = {str(Path(__file__)): sha256(__file__),
               str(ROOT / "src/videoqa_audit/blip_spatial_state.py"): sha256(ROOT / "src/videoqa_audit/blip_spatial_state.py")}
    for name in ("src/videoqa_methods/model_ops.py", "src/videoqa_runtime/baseline_selection.py", "src/videoqa_budget32/selection.py"):
        sources[str(ROOT / name)] = sha256(ROOT / name)
    inputs = {}
    # 步骤1：原问题是唯一模型文本；人工语义只写独立评价文件，不送入模型。
    for q, case in CASES.items():
        pool_path = BASE / "pools" / (q + ".json")
        pool = read_json(pool_path)
        sources[str(pool_path)] = sha256(pool_path)
        feature = BASE / "features" / (q + ".npy")
        sources[str(feature)] = sha256(feature)
        pairs = [(pool["candidates"][i], i) for i in case.get("coarse", [])]
        if "wfs" in case:
            path = PARENT / "panel" / q / "manifest.json"
            sources[str(path)] = sha256(path)
            bank = read_json(path)["inputs"]["REF-WFS-SB"]["frames"]
            pairs.extend((bank[i], None) for i in case["wfs"])
        pairs.sort(key=lambda x: x[0]["source_pts"])
        frames = [p[0] for p in pairs]
        coarse_indices = [p[1] for p in pairs]
        images, hashes = FrameProvider(dict(video=pool["video"], selected_frames=frames)).decode()
        dest = root / "inputs" / q
        dest.mkdir(parents=True, exist_ok=True)
        for i, image in enumerate(images):
            image.save(dest / f"{i}.png")
            image.close()
            sources[str(dest / f"{i}.png")] = sha256(dest / f"{i}.png")
        inputs[q] = dict(question_id=q, question=data[q]["question"], video=pool["video"], frames=frames,
                         rgb_sha256=hashes, coarse_indices=coarse_indices, feature_path=str(feature),
                         expected_scores=[pool["initial_scores"][i] if i is not None else None for i in coarse_indices])
        durable(root / "inputs" / (q + ".json"), inputs[q])
        sources[str(root / "inputs" / (q + ".json"))] = sha256(root / "inputs" / (q + ".json"))
    # 步骤2：模型只读哈希，现有环境及HF实现作为来源固定，不导入LAVIS运行时。
    for path in MODEL.iterdir():
        if path.is_file() and path.suffix in (".bin", ".json", ".txt", ".safetensors"):
            sources[str(path)] = sha256(path)
    from videoqa_budget32.selection import SOURCE
    old = read_json(SOURCE / "protocol.json")
    for q, item in inputs.items():
        path = item["video"]["path"]
        sources[path] = sha256(path)
        if sources[path] != old["video_hashes"][data[q]["video_id"]]:
            raise ValueError("Original source video identity changed")
    for path, item in old["model_files"].items():
        if path.startswith(str(MODEL) + "/") and sources.get(path) != item["sha256"]:
            raise ValueError("Original BLIP weight/processor identity changed")
    sources[str(SOURCE / "protocol.json")] = sha256(SOURCE / "protocol.json")
    import inspect
    from transformers.models.blip.modeling_blip_text import BlipTextSelfAttention
    from transformers.models.blip.modeling_blip import BlipForImageTextRetrieval
    for cls in (BlipTextSelfAttention, BlipForImageTextRetrieval):
        path = inspect.getfile(cls)
        sources[path] = sha256(path)
    durable(root / "evaluation.json", CASES)
    sources[str(root / "evaluation.json")] = sha256(root / "evaluation.json")
    spec = dict(run_id=root.name, inputs=inputs, sources=sources, actual_real_forward_image_pairs=30,
                real_forward_limit=32, synthetic_forward_limit=8, planned_synthetic_forward_image_pairs=4,
                qa_calls=0, block_index=6, batch_limit=16, fp32_tolerance=1e-6, utc=utc())
    durable(root / "protocol.json", spec)
    return spec


def verify(root):
    """接口：恢复逐个核验冻结字节，正在/失败的前向不自动重跑。"""
    spec = read_json(root / "protocol.json")
    for p, expected in spec["sources"].items():
        if sha256(p) != expected:
            raise ValueError("Frozen diagnostic source changed: " + p)
    return spec


def invoke(root, name, kind, scorer, encoded, images):
    """接口：开始前持久化调用，完整返回npz后标记完成；失败占用额度，禁止自动重试。"""
    import numpy as np
    path = root / "calls" / (name + "_" + kind + ".json")
    archive = path.with_suffix(".npz")
    identity = dict(protocol_sha256=sha256(root / "protocol.json"), name=name, kind=kind,
                    image_pairs=len(images), token_ids=encoded["input_ids"][0].tolist())
    if path.exists():
        record = read_json(path)
        if record.get("identity") != identity or record["status"] != "completed" or sha256(archive) != record["archive_sha256"]:
            raise ValueError("Uncertain BLIP call; repetition forbidden")
        with np.load(archive, allow_pickle=False) as saved:
            return {k: saved[k] for k in saved.files}, record
    spec = read_json(root / "protocol.json")
    existing = [read_json(p) for p in (root / "calls").glob("*.json")]
    synthetic = name == "synthetic"
    spent = sum(c["image_pairs"] for c in existing if (c["name"] == "synthetic") == synthetic)
    limit = spec["synthetic_forward_limit"] if synthetic else spec["real_forward_limit"]
    if spent + len(images) > limit:
        raise ValueError("Forward budget exhausted before model call")
    durable(path, dict(status="started_may_forward", name=name, kind=kind, identity=identity, image_pairs=len(images), utc=utc()))
    arrays, observation = spatial_batch(scorer, encoded, images, spatial=kind == "spatial")
    # 步骤1：前向返回立即保存，数据先于完成标记；不会把后处理失败当成重新调用理由。
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_suffix(".tmp.npz")
    np.savez(temporary, **arrays)
    os.replace(temporary, archive)
    record = dict(status="completed", name=name, kind=kind, image_pairs=len(images),
                  identity=identity, archive_sha256=sha256(archive), observation=observation, utc=utc())
    durable(path, record)
    return arrays, record


def run(root):
    """接口：单卡固定合成与四题控制/空间前向，无QA；不扫描层、词或视觉阈值。"""
    import numpy as np
    import torch
    from PIL import Image
    from videoqa_methods.model_ops import FeatureScorer
    from videoqa_budget32.selection import configure_visual_precision
    offline_environment()
    torch.set_num_threads(2)
    configure_visual_precision(torch)
    spec = verify(root)
    if (root / "results.json").exists():
        return read_json(root / "results.json")
    with exclusive(root / "run.lock"):
        scorer = FeatureScorer(MODEL)
        versions = dict(torch=torch.__version__, transformers=__import__("transformers").__version__)
        if (root / "environment.json").exists():
            old_env = read_json(root / "environment.json")
            if old_env["versions"] != versions or old_env["physical_gpu"] != os.environ.get("CUDA_VISIBLE_DEVICES"):
                raise ValueError("Recovery environment or physical GPU changed")
        durable(root / "environment.json", dict(versions=versions, gpu_name=torch.cuda.get_device_name(),
            physical_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"), pid=os.getpid(), sid=os.getsid(0),
            model_training=scorer.model.training, parameters_frozen=not any(p.requires_grad for p in scorer.model.parameters())))
        # 步骤1：固定两张合成图，共4个前向图文配对；不调用历史自动warmup。
        images = [Image.new("RGB", (384, 384), "gray"), Image.new("RGB", (384, 384), "white")]
        encoded = scorer.tokenize("Which synthetic color is visible?")
        control, a = invoke(root, "synthetic", "control", scorer, encoded, images)
        spatial, b = invoke(root, "synthetic", "spatial", scorer, encoded, images)
        if max(np.max(np.abs(control[k] - spatial[k])) for k in ("scores", "cls", "patches")) > 1e-6:
            raise ValueError("Synthetic interface changes original forward")
        result = {}
        # 步骤2：同批RGB先原控制再空间hook，核验ITM/CLS与历史以及接口一致性。
        for q, item in spec["inputs"].items():
            images = [Image.open(root / "inputs" / q / f"{i}.png").convert("RGB") for i in range(len(item["frames"]))]
            hashes = [hashlib.sha256(np.asarray(im).tobytes()).hexdigest() for im in images]
            if hashes != item["rgb_sha256"]:
                raise ValueError("Frozen input RGB changed")
            encoded = scorer.tokenize(item["question"])
            control, a = invoke(root, q, "control", scorer, encoded, images)
            spatial, b = invoke(root, q, "spatial", scorer, encoded, images)
            if a["observation"]["pixel_sha256"] != b["observation"]["pixel_sha256"]:
                raise ValueError("Same batch pixel identity differs")
            errors = {k: float(np.max(np.abs(control[k] - spatial[k]))) for k in ("scores", "cls", "patches")}
            if max(errors.values()) > 1e-6:
                raise ValueError("Attention hook changes original outputs")
            positions = [j for j, i in enumerate(item["coarse_indices"]) if i is not None]
            if positions:
                indices = [item["coarse_indices"][j] for j in positions]
                old = np.load(item["feature_path"], mmap_mode="r")[indices]
                errors["historical_scores"] = float(np.max(np.abs(control["scores"][positions] - [item["expected_scores"][j] for j in positions])))
                errors["historical_cls"] = float(np.max(np.abs(control["cls"][positions] - old)))
                if max(errors.values()) > 1e-6:
                    raise ValueError("Historical BLIP identity differs")
            measures = {}
            for label in ("positive", "negative"):
                i, j = CASES[q][label]
                measures[label] = compare_pair(spatial["cls"][i], spatial["cls"][j],
                    spatial["patches"][i], spatial["patches"][j], spatial["gradcam"][i], spatial["gradcam"][j])
            result[q] = dict(errors=errors, pairs=measures, positive_should_exceed_negative=True,
                             actual_tokens=len(encoded["input_ids"][0]))
            for image in images:
                image.close()
            durable(root / "status.json", dict(status="running", completed=len(result), total=4, utc=utc()))
        # 步骤3：汇总真实/合成图文配对与反向，最终问答0；空间图零质量不回退。
        calls = [read_json(p) for p in (root / "calls").glob("*.json")]
        real = sum(c["image_pairs"] for c in calls if c["name"] != "synthetic")
        synthetic = sum(c["image_pairs"] for c in calls if c["name"] == "synthetic")
        if real != 30 or real > spec["real_forward_limit"] or synthetic != 4:
            raise ValueError("Diagnostic call population differs")
        output = dict(status="completed", results=result, real_forward_image_pairs=real,
                      real_backward_image_pairs=15, synthetic_forward_image_pairs=synthetic, qa_calls=0,
                      all_model_parameters_frozen=not any(p.requires_grad for p in scorer.model.parameters()), utc=utc())
        durable(root / "results.json", output)
        print(output, flush=True)
        return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", choices=("prepare", "run"), default="prepare")
    args = parser.parse_args()
    if not args.run_id.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Invalid run ID")
    destination = ROOT / "outputs/diagnostics" / args.run_id
    {"prepare": prepare, "run": run}[args.stage](destination)
