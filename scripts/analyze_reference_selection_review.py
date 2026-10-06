"""WFS与RD全量只读对照、固定画面面板与来源完成审计；研究模型调用为0。"""
import os
os.environ.update(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
import argparse
from collections import Counter
import html
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from videoqa_full.state import durable, exclusive, rows, utc
from videoqa_runtime.common import read_json, sha256
from videoqa_audit.reference_selection_review import compare_source_frames, freeze_reference_panel
from analyze_coverage_objectives import DEV, Reader, image_board, load_pair

PARENT = ROOT / "outputs/analysis/coverage_objective_review_20261006_r3"
METHODS = ("BASE-Uniform", "BASE-BLIP-TopK", "RD-1.2-exp07", "RD-1.2-exp12", "REF-WFS-SB")


def analyze(root):
    """接口：复用已验收2700题来源，冻结双向得失与最多16题面板，不运行选择器。

    恢复核验已读取字节；输出存在时不重新选择案例或写回实验目录。
    """
    root.mkdir(parents=True, exist_ok=True)
    with exclusive(root / "analysis.lock"):
        if (root / "summary.json").exists():
            summary = read_json(root / "summary.json")
            for p, h in summary["sources"].items():
                if sha256(p) != h:
                    raise ValueError("Resume source changed: " + p)
            return summary
        reader = Reader()
        parent = reader.read(PARENT / "summary.json")
        audit = reader.read(PARENT / "completion_audit.json")
        if audit["status"] != "passed" or parent["new_model_calls"] != 0:
            raise ValueError("Parent audit is not complete")
        reader.bind(Path(__file__))
        reader.bind(ROOT / "src/videoqa_audit/reference_selection_review.py")
        data = rows()
        reader.bind(ROOT / "data/manifests/video_mme_full_2700.json")
        development = {r["video_id"] for r in reader.read(DEV)["rows"]}
        records = {}
        # 步骤1：逐条重读原始答案与源帧，检查父审计来源、题目和计分身份。
        for row in data:
            q = row["question_id"]
            original = reader.read(PARENT / "records/qwen32" / (q + ".json"))
            results = {}
            for method in METHODS:
                path = original["answer_sources"][method]
                item = reader.read(path)
                if reader.sources[path] != parent["sources"][path]:
                    raise ValueError("Historical answer changed: " + path)
                if (item["question"] != row["question"] or item["options"] != row["options"]
                        or item["question_id"] != q
                        or item["correct"] != (item["answer"]["parsed_answer"] == "ABCD"[row["answer_index"]])):
                    raise ValueError("Question or score identity changed")
                results[method] = item
            wfs = results["REF-WFS-SB"]
            flags = {m: results[m]["correct"] for m in METHODS}
            rd = flags["RD-1.2-exp12"]
            wf = flags["REF-WFS-SB"]
            cell = "both_correct" if rd and wf else "rd_only" if rd else "wfs_only" if wf else "both_wrong"
            record = dict(question_id=q, video_id=row["video_id"], stratum=row["stratum"],
                          task_type=row.get("task_type", "unknown"), correctness=flags,
                          predictions={m: results[m]["answer"]["parsed_answer"] for m in METHODS},
                          answer_sources=original["answer_sources"], rd12_wfs_cell=cell,
                          wfs_only_both_rd_wrong=wf and not rd and not flags["RD-1.2-exp07"],
                          both_rd_correct_wfs_wrong=rd and flags["RD-1.2-exp07"] and not wf,
                          same_input_wfs_rd12=compare_source_frames(results["RD-1.2-exp12"]["selection"], wfs["selection"]),
                          same_input_wfs_uniform=compare_source_frames(results["BASE-Uniform"]["selection"], wfs["selection"]))
            records[q] = record
            durable(root / "records" / (q + ".json"), record)
        # 步骤2：全量四格和时长/题型完整分母；结构差异只作为描述指标。
        groups = {}
        for key, subset in [("all2700", list(records.values()))] + [
                (s, [r for r in records.values() if r["stratum"] == s]) for s in ("short", "medium", "long")]:
            groups[key] = dict(n=len(subset), correct={m: sum(r["correctness"][m] for r in subset) for m in METHODS},
                              rd12_wfs_cells=dict(Counter(r["rd12_wfs_cell"] for r in subset)),
                              question_ids={c: [r["question_id"] for r in subset if r["rd12_wfs_cell"] == c]
                                            for c in ("both_correct", "both_wrong", "rd_only", "wfs_only")},
                              wfs_only_both_rd_wrong=[r["question_id"] for r in subset if r["wfs_only_both_rd_wrong"]],
                              both_rd_correct_wfs_wrong=[r["question_id"] for r in subset if r["both_rd_correct_wfs_wrong"]],
                              task_type_cells={t: dict(Counter(r["rd12_wfs_cell"] for r in subset if r["task_type"] == t))
                                               for t in sorted({r["task_type"] for r in subset})})
        panel = freeze_reference_panel(data, records, reader.read(PARENT / "panel.json"), development)
        durable(root / "panel.json", panel)
        summary = dict(status="completed", groups=groups, panel=panel, sources=reader.sources,
                       new_model_calls=0, new_answers=0, utc=utc(),
                       limitation="WFS has different source-time grid; exact source misses and temporal closeness are not semantic evidence judgments")
        durable(root / "summary.json", summary)
        print({k: {x: v[x] for x in ("n", "correct", "rd12_wfs_cells")} for k, v in groups.items()}, flush=True)
        print("Additional panel", panel["additional_slots"], flush=True)
        return summary


def render(root):
    """接口：CPU精确解码并核验官方Qwen预处理；生成WFS新增图板和原面板链接。

    只加载processor，不加载模型；RGB、提示Token和pixel字节须与正式输入相同。
    """
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoProcessor
    from videoqa_budget32.frames import FrameProvider
    from videoqa_budget32.observe import tensor_hash
    from videoqa_full.state import QWEN_PATH
    from videoqa_full.qwen_backend import MIN_PIXELS, MAX_PIXELS
    torch.set_num_threads(2)
    summary = read_json(root / "summary.json")
    parent_sources = read_json(PARENT / "summary.json")["sources"]
    data = {r["question_id"]: r for r in rows()}
    processor = AutoProcessor.from_pretrained(QWEN_PATH, local_files_only=True, use_fast=False,
        min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS,
        size={"shortest_edge": MIN_PIXELS, "longest_edge": MAX_PIXELS})
    assets = {}
    # 步骤1：原12题已有Qwen图板则引用；新增题及LLaVA专属旧题补齐同配置画面。
    for case in summary["panel"]["questions"]:
        q = case["question_id"]
        record = read_json(root / "records" / (q + ".json"))
        dest = root / "panel" / q
        dest.mkdir(parents=True, exist_ok=True)
        manifests = {}
        for method in METHODS:
            path = record["answer_sources"][method]
            if sha256(path) != summary["sources"][path]:
                raise ValueError("Render source changed")
            item = read_json(path)
            images, hashes = FrameProvider(item["selection"]).decode()
            if hashes != item["rgb_sha256"]:
                raise ValueError("Actual RGB changed")
            observed = item["answer"]["observed"]
            context = observed["context"]
            messages = [dict(role="system", content="You are a helpful assistant."),
                        dict(role="user", content=[dict(type="video"), dict(type="text", text=context)])]
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            pixels = processor(text=[prompt], videos=[np.stack([np.asarray(im) for im in images])], fps=2.0,
                images_kwargs={"size": {"shortest_edge": MIN_PIXELS, "longest_edge": MAX_PIXELS}},
                padding=False, truncation=False, return_tensors="pt")
            pixel_hash = tensor_hash(pixels["pixel_values_videos"])
            if pixel_hash != observed["model_pixel_sha256"] or pixels["input_ids"][0].tolist() != observed["input_token_ids"]:
                raise ValueError("Official preprocessing changed")
            # 步骤2：原尺寸与模型尺寸并列，前缀/配额作为追踪，不自动给语义标签。
            index = METHODS.index(method)
            h, w = observed["processed_size"]
            model_views = [im.resize((w, h), Image.Resampling.BICUBIC) for im in images]
            frames = item["selection"]["selected_frames"]
            image_board(images, frames, dest / f"{index}_raw.jpg")
            image_board(model_views, frames, dest / f"{index}_model.jpg")
            for i, (im, view) in enumerate(zip(images, model_views)):
                im.save(dest / f"{index}_{i:02d}_raw.png")
                view.save(dest / f"{index}_{i:02d}_model.png")
                im.close()
                view.close()
            manifests[method] = dict(answer_source=path, answer_sha256=summary["sources"][path],
                rgb_sha256=hashes, processor_sha256=pixel_hash, frames=frames,
                raw_board=f"{index}_raw.jpg", model_board=f"{index}_model.jpg")
        reader = Reader()
        pool, a, b = load_pair("qwen32", q, reader)
        for p, expected in reader.sources.items():
            # 只有父来源中已冻结资产可被引用；这一步不重算特征或选帧。
            if parent_sources.get(p) != expected:
                raise ValueError("Pool source changed")
        durable(dest / "manifest.json", dict(inputs=manifests, plan=pool["plan"],
            prefix07=a, prefix12=b, sources=reader.sources, new_model_calls=0))
        assets[q] = dict(manifest_sha256=sha256(dest / "manifest.json"), source_count=len(manifests))
        row = data[q]
        page = ["<!doctype html><meta charset='utf-8'><style>body{max-width:1400px;margin:auto;font:16px sans-serif}img{max-width:100%}.names{display:none}</style>",
                "<h1>" + q + " / Qwen32</h1><p>" + html.escape(row["question"]) + "</p><p>" + html.escape(" / ".join(row["options"])) + "</p>",
                "<p>事后审核；先看画面，再展开方法。邻近时间不等于相同证据。</p>",
                "<button onclick=\"document.querySelectorAll('.names').forEach(x=>x.style.display='block')\">展开方法与回答</button>"]
        for index, method in enumerate(METHODS):
            page += [f"<h2>输入{index + 1}</h2><div class='names'>" + html.escape(method + " / " + str(record["predictions"][method]) + " / " + str(record["correctness"][method])) + "</div>",
                     f"<img src='{index}_raw.jpg'><details><summary>模型尺寸视图与逐帧链接</summary><img src='{index}_model.jpg'>",
                     " ".join(f"<a href='{index}_{j:02d}_raw.png'>原帧{j+1}</a> <a href='{index}_{j:02d}_model.png'>输入视图</a>" for j in range(len(manifests[method]["frames"]))) + "</details>"]
        page += ["<details><summary>完整区域、局部前缀与配额</summary><a href='manifest.json'>manifest.json</a></details>"]
        (dest / "index.html").write_text("\n".join(page), encoding="utf-8")
        print("Rendered", q, flush=True)
    # 步骤3：独立索引与审核入口，用户留言不冒充AI判断或一致率。
    index = ["<!doctype html><meta charset='utf-8'><h1>WFS与RD固定参考复核</h1><p>零模型调用；12题复用、最多4题双向新增。</p>"]
    for case in summary["panel"]["questions"]:
        q = case["question_id"]
        index.append(f"<p><a href='panel/{q}/index.html'>{q}</a> " + html.escape(case["slot"]) + "</p>")
    (root / "index.html").write_text("\n".join(index), encoding="utf-8")
    durable(root / "render.json", dict(status="completed", records=assets, new_model_calls=0, utc=utc()))


def audit(root):
    """接口：核验全部来源、2700记录、固定画面输入及语义初审记录闭合，0模型调用。

    完成后恢复只读取，生成账本不可由本入口访问或新增。
    """
    summary = read_json(root / "summary.json")
    rendered = read_json(root / "render.json")
    review = read_json(root / "review.json")
    if len(list((root / "records").glob("*.json"))) != 2700:
        raise ValueError("Incomplete full population")
    questions = {x["question_id"] for x in summary["panel"]["questions"]}
    if questions != {x["question_id"] for x in review["cases"]} or questions != set(rendered["records"]):
        raise ValueError("Panel review is incomplete")
    for p, h in summary["sources"].items():
        if sha256(p) != h:
            raise ValueError("Historical source changed: " + p)
    for q, item in rendered["records"].items():
        path = root / "panel" / q / "manifest.json"
        if sha256(path) != item["manifest_sha256"]:
            raise ValueError("Rendered identity changed")
        for p, h in read_json(path)["sources"].items():
            if sha256(p) != h:
                raise ValueError("Pool or feature source changed")
    durable(root / "completion_audit.json", dict(status="passed", records=2700,
        reviewed_questions=len(questions), new_model_calls=0,
        historical_sources_unchanged=len(summary["sources"]), utc=utc()))


def main():
    """接口：只读分析/CPU画面/完成审计分阶段运行，不具备问答执行权限或模型加载路径。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", choices=("analyze", "render", "audit"), default="analyze")
    args = parser.parse_args()
    if not args.run_id.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Invalid run ID")
    root = ROOT / "outputs/analysis" / args.run_id
    {"analyze": analyze, "render": render, "audit": audit}[args.stage](root)


if __name__ == "__main__":
    main()
