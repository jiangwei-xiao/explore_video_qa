"""参考选帧的只读时间分布对照；邻近时间不等同于语义证据。"""
from fractions import Fraction


def compare_source_frames(left, right):
    """接口：比较同视频两份冻结输入，返回精确重合及时间分布，禁止跨时钟匹配。

    输入是带video和selected_frames的记录；返回仅含结构指标，不推断证据充分性。
    """
    # 步骤1：源身份必须属于同一视频和时钟，候选编号不参与跨方法对应。
    for key in ("path", "time_base", "start_pts"):
        if left["video"][key] != right["video"][key]:
            raise ValueError("Video identity differs: " + key)
    clock = Fraction(left["video"]["time_base"])
    origin = left["video"]["start_pts"]

    def describe(selection):
        frames = selection["selected_frames"]
        keys = [(f["source_frame_index"], f["source_pts"]) for f in frames]
        if not keys or len(set(keys)) != len(keys):
            raise ValueError("Empty or duplicate source input")
        times = [(p - origin) * clock for _, p in keys]
        if times != sorted(times):
            raise ValueError("Unordered source input")
        gaps = [b - a for a, b in zip(times, times[1:])]
        return keys, times, dict(
            count=len(frames), span_seconds=float(times[-1] - times[0]),
            adjacent_pairs_within_one_second=sum(g <= 1 for g in gaps),
            maximum_gap_seconds=float(max(gaps, default=Fraction(0))),
        )

    # 步骤2：精确重合与邻近观察分开；0.5秒只描述距离，不宣称等价源帧或事实命中。
    lk, lt, lm = describe(left)
    rk, rt, rm = describe(right)
    return dict(
        exact_shared_sources=len(set(lk) & set(rk)),
        identical_ordered_sources=lk == rk,
        left=lm, right=rm,
        right_nearest_left_seconds=[float(min(abs(t - s) for s in lt)) for t in rt],
        right_temporally_near_left_half_second=sum(
            min(abs(t - s) for s in lt) <= Fraction(1, 2) for t in rt
        ),
        limitation="Temporal proximity is descriptive; visual and relational evidence require review",
    )


def freeze_reference_panel(data, records, previous_panel, development_videos):
    """接口：保留原12题，再固定短/中两个时长层的双向得失，共最多4道新题。

    开发视频外优先、原冻结题序、视频不重复，歧义题排除；空槽位不跨层补齐。
    """
    # 步骤1：已有认识不重新挑题，所有题使用Qwen32同后端参考对照。
    by_id = {row["question_id"]: row for row in data}
    panel = [dict(question_id=x["question_id"], slot="existing_" + x["slot"],
                  previously_reviewed=True) for x in previous_panel["questions"]]
    used = {by_id[x["question_id"]]["video_id"] for x in panel}
    ordered = sorted(enumerate(data), key=lambda x: (
        x[1]["video_id"] in development_videos, x[0]))
    slots = []
    # 步骤2：两方向均取案例，不依据画面审核后的结论换题。
    for stratum in ("short", "medium"):
        for direction in ("wfs_only_both_rd_wrong", "both_rd_correct_wfs_wrong"):
            q = next((r["question_id"] for _, r in ordered
                      if r["stratum"] == stratum and r["video_id"] not in used
                      and r["question_id"] not in ("491-2", "321-2")
                      and records[r["question_id"]][direction]), None)
            slot = stratum + "_" + direction
            slots.append(dict(slot=slot, question_id=q))
            if q is not None:
                panel.append(dict(question_id=q, slot=slot, previously_reviewed=False))
                used.add(by_id[q]["video_id"])
    return dict(questions=panel, additional_slots=slots, maximum_new_questions=4,
                selection_rule="outside development videos first; frozen order; distinct videos; empty slots retained")
