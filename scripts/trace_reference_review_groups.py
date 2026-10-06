"""人工已确认局部状态的源路径追踪，只供评价，不传入任何选帧函数。"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from videoqa_audit.coverage_objective_review import frame_identity, trace_witness_path
from videoqa_full.state import durable
from analyze_coverage_objectives import Reader, load_pair


def trace_groups(root, definitions):
    """接口：对人工独立登记的粗帧组核验前缀位置和配额，不扩大75事实主分母。

    definitions为题号、说明与已查看的coarse_indices；输出单帧路径及联合可达条件。
    """
    root = Path(root)
    reader = Reader()
    reader.bind(Path(__file__))
    reader.bind(ROOT / "src/videoqa_audit/coverage_objective_review.py")
    records = []
    # 步骤1：粗编号只在对应视频的初始池内解释，随后转换真实源身份。
    for definition in definitions:
        q = definition["question_id"]
        pool, left, right = load_pair("qwen32", q, reader)
        indices = definition["coarse_indices"]
        if len(indices) != len(set(indices)) or any(not 0 <= i < pool["initial_count"] for i in indices):
            raise ValueError("Invalid independent coarse witness group")
        keys = [frame_identity(pool["candidates"][i]) for i in indices]
        item = dict(**definition, frames=[pool["candidates"][i] for i in indices])
        # 步骤2：单帧可达不等于联合可达，组所需各区域最小配额须同时满足总预算。
        for name, trace in (("exp07", left), ("exp12", right)):
            route = trace_witness_path(pool, trace, keys)
            minima = {str(r["region_id"]): 1 for r in trace["regions"]}
            in_prefix = all(p.get("prefix_position") is not None for p in route["paths"])
            if in_prefix:
                for p in route["paths"]:
                    rid = str(p["region_id"])
                    minima[rid] = max(minima[rid], p["prefix_position"])
            budget = sum(trace["region_budgets"].values())
            item[name] = dict(paths=route, group_fully_selected=all(p["selected"] for p in route["paths"]),
                group_in_prefix=in_prefix, minimum_quotas=minima if in_prefix else None,
                group_reachable_by_quota_only=in_prefix and sum(minima.values()) <= budget,
                warning="已列状态组合的结构条件；不证明整题证据充分或其他区域转移无损")
        records.append(item)
    # 步骤3：仅写独立审计目录，来源指纹可恢复核验，正式问答调用0。
    durable(root / "independent_group_paths.json", dict(records=records, sources=reader.sources,
        main_proxy_denominator_unchanged=dict(questions=42, facts=75), new_model_calls=0))
    return records


def render_known_groups(root, records):
    """接口：精确解码已登记的独立粗帧见证供画面核验，不创建正式采样候选。

    只在审计目录输出原帧和图板；源帧身份与时间来自保存池，不按平均FPS计算。
    """
    from videoqa_budget32.frames import FrameProvider
    from analyze_coverage_objectives import image_board
    # 步骤1：只读取已登记源帧，转换图像不会触发视觉编码或问答。
    for record in records:
        q = record["question_id"]
        pool, _, _ = load_pair("qwen32", q, Reader())
        images, hashes = FrameProvider(dict(video=pool["video"], selected_frames=record["frames"])).decode()
        dest = Path(root) / "panel" / q / "independent"
        dest.mkdir(parents=True, exist_ok=True)
        image_board(images, record["frames"], dest / "coarse_states.jpg")
        # 步骤2：保存真实像素来源，图板只是审核视图，不反向更改实验池。
        for index, image in zip(record["coarse_indices"], images):
            image.save(dest / f"coarse_{index}.png")
            image.close()
        durable(dest / "manifest.json", dict(frames=record["frames"], rgb_sha256=hashes,
                                             new_model_calls=0, experimental_pool_modified=False))


def measure_flatness(image):
    """接口：只度量全画布RGB变化；每通道范围最多2个灰度级标为近纯色疑点。

    不裁剪、不做暗场阈值、不判断语义；有文字/对象的黑白背景通常具有较大范围。
    """
    import numpy as np
    pixels = np.asarray(image.convert("RGB"), dtype=np.uint8).reshape(-1, 3)
    ranges = np.ptp(pixels, axis=0).astype(int).tolist()
    return dict(channel_ranges=ranges, maximum_channel_range=max(ranges),
                nearly_constant_canvas=max(ranges) <= 2)


def frame_validity(root):
    """接口：固定16题已生成画面的纯色疑点普查，分母是审核面板而非全2700题。

    只使用PNG像素；标记不是过滤结果，不把纯色与答题无关自动等同。
    """
    from PIL import Image
    from videoqa_runtime.common import read_json, sha256
    records = []
    sources = {str(Path(__file__)): sha256(__file__)}
    # 步骤1：读取正式画面清单及原尺寸/模型尺寸双视图，不运行评分或重新选择。
    for case in read_json(Path(root) / "panel.json")["questions"]:
        q = case["question_id"]
        dest = Path(root) / "panel" / q
        manifest = read_json(dest / "manifest.json")
        for index, (method, item) in enumerate(manifest["inputs"].items()):
            for ordinal, frame in enumerate(item["frames"]):
                views = {}
                for kind in ("raw", "model"):
                    path = dest / f"{index}_{ordinal:02d}_{kind}.png"
                    sources[str(path)] = sha256(path)
                    with Image.open(path) as image:
                        views[kind] = measure_flatness(image)
                records.append(dict(question_id=q, method=method, ordinal=ordinal, frame=frame, views=views,
                    flat_in_both_views=all(v["nearly_constant_canvas"] for v in views.values())))
    # 步骤2：统计只是候选有效性的可见疑点，语义核验另行保留，不产生新16/32帧输入。
    durable(Path(root) / "frame_validity.json", dict(records=records, sources=sources,
        criterion="max per-channel pixel range <= 2/255, full canvas; raw and model views must both satisfy",
        panel_questions=len({r["question_id"] for r in records}), new_model_calls=0,
        semantic_invalidity_not_inferred=True, algorithm_modified=False))


def inspect_components(root):
    """接口：复用缓存CLS重算已定位状态在固定局部前缀每一步的支持与重复度。

    同时保存真正入选的当步画面，证明排序路径，不把分数推导成语义置信度。
    """
    import numpy as np
    from videoqa_runtime.common import read_json
    from videoqa_audit.coverage_objective_review import component_at
    from analyze_coverage_objectives import BASE
    reader = Reader()
    reader.bind(Path(__file__))
    groups = read_json(Path(root) / "independent_group_paths.json")["records"]
    results = []
    # 步骤1：只有已实际核查的粗见证被用于解释，不把人工目标传给选择器。
    for group in groups:
        q = group["question_id"]
        pool, trace, _ = load_pair("qwen32", q, reader)
        path = reader.bind(BASE / "features" / (q + ".npy"))
        features = np.load(path, allow_pickle=False)
        indices = set(group["coarse_indices"])
        for region in trace["regions"]:
            targets = sorted(indices.intersection(region["members"]))
            if not targets:
                continue
            before = []
            steps = []
            # 步骤2：按真实保存的选择顺序计算；同时给出最大相似的已选源帧。
            for index in region["prefix"]:
                proposed = []
                for target in targets:
                    same = max(before, key=lambda j: float(features[target] @ features[j])) if before else None
                    proposed.append(dict(frame=pool["candidates"][target],
                        **component_at(pool, features, region, target, before),
                        closest_selected_frame=pool["candidates"][same] if same is not None else None))
                steps.append(dict(chosen_frame=pool["candidates"][index],
                    chosen_components=component_at(pool, features, region, index, before), targets=proposed))
                before.append(index)
            results.append(dict(question_id=q, region_id=region["region_id"], steps=steps,
                                limitation="appearance support/redundancy only; semantic value comes from independent review"))
    # 步骤3：输出独立数值复盘，不改配额、局部前缀或生成任何新答案。
    durable(Path(root) / "component_paths.json", dict(records=results, sources=reader.sources, new_model_calls=0))


if __name__ == "__main__":
    destination = ROOT / "outputs/analysis/reference_selection_review_20261006_r1"
    if "--frame-validity" in sys.argv:
        frame_validity(destination)
        sys.exit(0)
    if "--components" in sys.argv:
        inspect_components(destination)
        sys.exit(0)
    traced = trace_groups(destination, [
        dict(question_id="409-3", description="持盒与盒子打开取枪两个已定位状态", coarse_indices=[77, 80]),
        dict(question_id="556-1", description="离家、滑板可见与学校建筑", coarse_indices=[41, 42, 43]),
        dict(question_id="363-2", description="玩车、枪出现与人物逃离", coarse_indices=[71, 74, 76]),
        dict(question_id="035-2", description="机构列表逐行书写状态，独立补充审核", coarse_indices=[55, 56, 59]),
    ])
    render_known_groups(destination, traced)
