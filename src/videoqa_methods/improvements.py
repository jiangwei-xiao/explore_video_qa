"""首轮改进：局部受限重选及原词受限查询；不修改旧方法入口。"""
import collections
import copy
from fractions import Fraction
import re
import time
import numpy as np

from videoqa_runtime.baseline_selection import ProtocolError
from videoqa_runtime.video import decode_selected
from .algorithm import best, check_features, membership, segment_scores, compete, hotspot_candidates
from .pipeline import scan
from .refinement import decode_hotspots

QUERY_PROMPT = '''Rewrite the question as one English visual-evidence search description.
Do not answer the question or infer any unknown fact.
Reuse the original content words exactly. You may reorder them.
Preserve named entities, numbers, negation, temporal qualifiers,
comparison qualifiers, and their relationships.
Only add or remove function words and generic evidence words.
Return only the description, without a prefix or explanation.

Question:
{question}'''
WHITELIST = frozenset("""a an the this that these those it its he his she her they their
of in on at to for from with by and or as
is are was were be been being do does did
can could will would should have has had
what which who whom whose how why what's who's
according based given following presented provided shown
video information evidence object action event person people""".split())


def reselect_local(pool, features, cfg):
    """接口：输入冻结粗选/补查池与纯视觉特征，返回16帧索引和逐段追踪；仅窗口内预算可以重新竞争。"""
    # 步骤1：用有理数PTS确定窗口并集，固定片段归属及原配额。
    rows, r = pool['candidates'], np.asarray(pool['scores'], dtype=np.float64)
    f = check_features(features, len(rows))
    groups = membership(rows, pool['segments'])
    windows = [(Fraction(w['left_fraction']), Fraction(w['right_fraction']))
               for w in pool['hotspot_trace'] if w['selected']]
    tb, origin = Fraction(pool['video']['time_base']), pool['video']['start_pts']
    inside = [any(a <= (c['source_pts'] - origin) * tb <= b for a, b in windows) for c in rows]
    changed = set(groups[pool['initial_count']:].tolist())
    selected, trace = [], []
    for sid, quota in enumerate(pool['quotas']):
        coarse = [i for i in pool['coarse_selected'] if groups[i] == sid]
        fixed = [i for i in coarse if not inside[i]]
        k = len(coarse) - len(fixed)
        eligible = [i for i in range(len(rows)) if groups[i] == sid and inside[i]]
        rec = dict(segment_id=sid, quota=quota, fixed_indices=fixed, replaceable_slots=k,
                   eligible_indices=eligible, received_new=sid in changed, steps=[])
        # 步骤2：无新增/零配额/零窗口内名额均保持粗选，不制造额外预算。
        if not quota or sid not in changed or not k:
            selected.extend(coarse)
            rec.update(reason='unchanged', selected_indices=coarse)
            trace.append(rec)
            continue
        local, remaining = list(fixed), set(eligible)
        if len(remaining) < k:
            raise ProtocolError('Window pool cannot fill original local quota')
        # 步骤3：固定帧和已选新帧共同作为同段去重参照，负增益仍填满原名额。
        for step in range(k):
            dup = np.zeros(len(rows)) if not local else np.clip(f @ f[local].T, 0, 1).max(axis=1)
            gains = ((1 - pool['lambda_value']) * r + pool['lambda_value'] *
                     pool['segments'][sid]['representative_score'] / (1 + len(local)) -
                     cfg['redundancy_coefficient'] * dup)
            i = best(remaining, gains, rows)
            rec['steps'].append(dict(step=step, candidate_index=i, gain=float(gains[i]),
                                     relevance=float(r[i]), duplicate=float(dup[i]),
                                     segment_count_before=len(local), redundancy_reference_indices=list(local)))
            local.append(i)
            remaining.remove(i)
        rec.update(reason='local_reselection', selected_indices=local)
        trace.append(rec)
        selected.extend(local)
    # 步骤4：恢复真实时间顺序并检查唯一帧、配额和窗口外保护不变量。
    selected.sort(key=lambda i: (rows[i]['source_pts'], rows[i]['candidate_index']))
    if len(selected) != cfg['frames'] or len(set(rows[i]['source_pts'] for i in selected)) != cfg['frames']:
        raise ProtocolError('C-local must preserve 16 unique source frames')
    if np.bincount(groups[selected], minlength=len(pool['segments'])).tolist() != pool['quotas']:
        raise ProtocolError('C-local changed segment quotas')
    if not {i for i in pool['coarse_selected'] if not inside[i]} <= set(selected):
        raise ProtocolError('C-local removed an outside-window coarse frame')
    return selected, trace


def select_clocal(path, question, scorer, scope, cfg, progress=None):
    """接口：已完成一次分类后，从原视频独立扫描到最终取帧；不调用旧C的最终重选、取帧或问答。"""
    # 步骤1：复用原C的扫描、分段及粗选函数，独立计算本题池。
    pool, features, times, encoded = scan(path, question, scorer, progress)
    lam = cfg['lambda'][scope['label']]
    t = time.perf_counter()
    segments, wavelet = segment_scores(pool['scores'], pool['candidates'], pool['video'], cfg)
    times['segmentation_seconds'] = time.perf_counter() - t
    t = time.perf_counter()
    coarse, quotas, competition = compete(pool['candidates'], pool['scores'], features, segments, lam, cfg, cfg['frames'])
    times['competition_seconds'] = time.perf_counter() - t
    n = len(pool['candidates'])
    # 步骤2：按原规则生成热点、精确补查和同一次BLIP视觉特征。
    t = time.perf_counter()
    hotspots, windows = hotspot_candidates(pool['candidates'], pool['scores'], features, coarse, segments, quotas, pool['video'], cfg)
    times['hotspot_seconds'] = time.perf_counter() - t
    added, images, decoding, fine_times = decode_hotspots(pool['video'], pool['candidates'], windows, cfg)
    times.update(fine_times, fine_preprocess_seconds=0.0, fine_forward_features_seconds=0.0)
    blocks = [features]
    for i in range(0, len(images), 16):
        values, matrix, pre, forward = scorer.score_batch_features(encoded, images[i:i + 16])
        pool['scores'].extend(values)
        blocks.append(matrix)
        times['fine_preprocess_seconds'] += pre
        times['fine_forward_features_seconds'] += forward
    pool['candidates'].extend(added)
    features = np.concatenate(blocks)
    pool.update(scope=scope, lambda_value=lam, initial_count=n, segments=segments, wavelet=wavelet,
                segment_ids=membership(pool['candidates'], segments).tolist(), coarse_selected=coarse,
                quotas=quotas, competition_trace=competition, hotspot_trace=hotspots, fine_decode_trace=decoding)
    # 步骤3：唯一改动是局部受限重选，然后精确解码最终16帧。
    t = time.perf_counter()
    selected, trace = reselect_local(pool, features, cfg)
    times['reselection_seconds'] = time.perf_counter() - t
    pool.update(selected_indices=selected, selected_frames=[pool['candidates'][i] for i in selected], local_reselection_trace=trace)
    t = time.perf_counter()
    frames = decode_selected(pool['video'], pool['selected_frames'])
    times['selected_decode_seconds'] = time.perf_counter() - t
    times['selection_core_seconds'] = sum(v for k, v in times.items() if k not in
                                         ('candidate_decode_seconds', 'fine_decode_seconds', 'selected_decode_seconds'))
    return frames, pool, features, times


def lexical_tokens(text):
    """接口：按冻结规则提取大小写无关英文词/数字，保留撇号及所有数字出现次数。"""
    return re.findall(r"[a-z]+(?:'[a-z]+)*|[0-9]+", text.replace('’', "'").replace('‘', "'").lower())


def validate_query(question, raw, reached_limit=False):
    """接口：检查原词多重集合及异常格式；失败返回原问题，不生成第二版。词汇通过不等于语义保证。"""
    # 步骤1：拒绝空串、控制/解释/答案格式和非正常长度结束。
    query, reasons = raw.strip(), []
    if not query: reasons.append('empty')
    if '\n' in query or '\r' in query: reasons.append('multiline')
    if reached_limit: reasons.append('generation_limit_without_eos')
    if re.search(r'<[^>]*>|\[[A-D]\]|[\x00-\x08\x0b\x0c\x0e-\x1f]', query): reasons.append('control_marker')
    if re.search(r'(^|\s)[A-D][.)](\s|$)|^(answer|description|query|explanation)\s*:', query, re.I):
        reasons.append('answer_or_prefix')
    # 步骤2：非白名单词必须逐次保留；数字、否定、先后和比较词不在白名单中。
    before = collections.Counter(t for t in lexical_tokens(question) if t not in WHITELIST)
    after = collections.Counter(t for t in lexical_tokens(query) if t not in WHITELIST)
    if before != after: reasons.append('content_multiset_mismatch')
    return dict(query=question if reasons else query, fallback=bool(reasons), reasons=reasons,
                added_content=list((after - before).elements()), missing_content=list((before - after).elements()),
                semantic_fidelity='pending_review' if not reasons else 'original_question_fallback')


def rewrite_query(question, backend, state):
    """接口：一次纯文本受限查询生成，返回原始输出、校验/回退、Token及同步耗时；state由外部持久化调用账本保护。"""
    import torch
    from transformers import GenerationConfig
    from llava.conversation import conv_templates
    # 步骤1：建立独立qwen_1_5会话和局部128Token配置，不污染8Token问答。
    start = time.perf_counter()
    conv = copy.deepcopy(conv_templates['qwen_1_5'])
    conv.append_message(conv.roles[0], QUERY_PROMPT.format(question=question))
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    tok, model = backend.tokenizer, backend.model
    inputs = tok(prompt, return_tensors='pt', truncation=False).to('cuda')
    if inputs['input_ids'].shape[1] + 128 > backend.context:
        raise ProtocolError('Query exceeds context')
    cfg = GenerationConfig(max_new_tokens=128, do_sample=False, num_beams=1, use_cache=True,
                           bos_token_id=model.generation_config.bos_token_id, eos_token_id=tok.eos_token_id,
                           pad_token_id=tok.pad_token_id or tok.eos_token_id)
    # 步骤2：一次调用后记录Token；达到上限但无EOS视为失败回退。
    state['started'] = True
    with torch.inference_mode():
        output = model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], generation_config=cfg)
    state['returned'] = True
    torch.cuda.synchronize()
    ids = output[0].tolist()
    raw = tok.batch_decode(output, skip_special_tokens=True)[0].strip()
    result = dict(raw_output=raw, **validate_query(question, raw, len(ids) >= 128 and tok.eos_token_id not in ids),
                  prompt=prompt, input_tokens=int(inputs['input_ids'].shape[1]), output_token_ids=ids,
                  seconds=time.perf_counter() - start, visual_inputs=0, generation_max_new_tokens=128)
    state['result'] = result
    return result


def compare_upstream(pool, features, historical, historical_features):
    """接口：检查C-local与冻结C上游一致性；仅忽略耗时，离散选择必须完全一致。"""
    failures = []
    for key in ('candidates', 'initial_count', 'segment_ids', 'coarse_selected', 'quotas', 'lambda_value'):
        if pool[key] != historical[key]: failures.append(key)
    if pool['scope']['label'] != historical['scope']['label']: failures.append('scope_label')
    for key in ('scores',):
        if np.shape(pool[key]) != np.shape(historical[key]) or not np.allclose(pool[key], historical[key], atol=1e-6, rtol=0):
            failures.append(key)
    if features.shape != historical_features.shape or not np.allclose(features, historical_features, atol=1e-6, rtol=0):
        failures.append('features')
    # 步骤2：热点的浮点诊断量允许同样数值误差，窗口身份、原因和选择标记要求一致。
    def equivalent(a, b):
        """递归核对结构、离散值和浮点误差，不把时间性能记录混入身份判断。"""
        if isinstance(a, dict): return isinstance(b, dict) and a.keys() == b.keys() and all(equivalent(a[k], b[k]) for k in a)
        if isinstance(a, list): return isinstance(b, list) and len(a) == len(b) and all(equivalent(x, y) for x, y in zip(a, b))
        if isinstance(a, float): return isinstance(b, (float, int)) and abs(a - b) <= 1e-6
        return a == b
    if not equivalent(pool['hotspot_trace'], historical['hotspot_trace']): failures.append('hotspots')
    if not equivalent(pool['segments'], historical['segments']): failures.append('segments')
    return dict(passed=not failures, failures=failures, numeric_atol=1e-6, discrete_exact=True)
