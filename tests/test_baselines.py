from fractions import Fraction
import json
import numpy as np
import pytest
from videoqa_runtime.baseline_selection import select_topk, iter_candidate_frames, prepare_selection, ProtocolError
from videoqa_runtime.video import candidate_rows, decode_selected
from videoqa_runtime.baseline_records import check_recovery
from videoqa_runtime.baseline_report import wilson, paired_interval


def test_topk_ties_follow_source_time_then_id():
    rows = [dict(candidate_index=i, source_pts=t) for i, t in enumerate([30, 10, 20, 40])]
    assert [r['source_pts'] for r in select_topk(rows, [.8, .8, .8, .9], 3)] == [10, 20, 40]


@pytest.mark.parametrize('scores', [[float('nan')]*16, [1.01]*16, [-.1]*16, [float('inf')]*16])
def test_invalid_scores_stop(scores):
    with pytest.raises(ProtocolError):
        select_topk([dict(candidate_index=i, source_pts=i) for i in range(16)], scores)


def test_short_pool_stops():
    with pytest.raises(ProtocolError):
        select_topk([dict(candidate_index=0, source_pts=0)], [.5])


def test_uncertain_attempt_cannot_be_silently_retried(tmp_path):
    p = tmp_path/'attempts/q'; p.mkdir(parents=True)
    (p/'uniform.01.json').write_text(json.dumps(dict(question_id='q', method='uniform', status='running_may_generate')))
    with pytest.raises(ProtocolError, match='Uncertain'):
        check_recovery(tmp_path, [], 'protocol')


def test_wilson_and_paired_bootstrap_boundaries():
    assert wilson(0, 50)[0] == pytest.approx(0)
    assert abs(wilson(50, 50)[1] - 1) < 1e-12
    rows = [dict(question_id=str(i), stratum=s) for i, s in enumerate(['short','medium','long'])]
    records = {m: {str(i): {'correct': True} for i in range(3)} for m in ['uniform','topk']}
    assert paired_interval(rows, records, 100) == [0., 0.]


def test_streaming_decoder_matches_pts_protocol_and_bounds_batches(tmp_path):
    import av
    p = tmp_path/'synthetic.mp4'
    with av.open(str(p), 'w') as output:
        stream = output.add_stream('mpeg4', rate=4)
        stream.width, stream.height, stream.pix_fmt = 64, 48, 'yuv420p'
        for i in range(80):
            frame = av.VideoFrame.from_ndarray(np.full((48,64,3), i, np.uint8), format='rgb24')
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    meta = {}
    pairs = list(iter_candidate_frames(p, meta))
    with av.open(str(p)) as container:
        stream = container.streams.video[0]
        original = list(candidate_rows([f.pts for f in container.decode(stream)], Fraction(stream.time_base), stream.start_time or 0))
    for i, row in enumerate(original): row['candidate_index'] = i
    assert [r for r, _ in pairs] == original

    class Scorer:
        def __init__(self): self.batches = []
        def tokenize(self, question): return {'input_ids': np.zeros((1, 10))}
        def score_batch(self, encoded, images):
            self.batches.append(len(images))
            return [float(np.asarray(im).mean()) / 255 for im in images], 0., 0.
    scorer = Scorer()
    uniform, u = prepare_selection(p, 'Synthetic?', 'uniform', scorer)
    assert scorer.batches == []
    topk, t = prepare_selection(p, 'Synthetic?', 'topk', scorer)
    assert scorer.batches == [16, 4]
    assert u['candidates'] == t['candidates'] and len(topk) == len(uniform) == 16
    assert len({r['source_pts'] for r in t['selected_frames']}) == 16


def test_saved_result_recovers_without_repeating_uncertain_journal(tmp_path):
    row = dict(question_id='q', video_id='v', question='Question?', options=['A. a','B. b','C. c','D. d'], answer_index=0)
    candidates = [dict(candidate_index=i, source_pts=i) for i in range(16)]
    record = dict(status='completed', protocol_sha256='p', question_id='q', video_id='v', question=row['question'],
                  options=row['options'], method='uniform', application_cache_hits=0, generation_attempts=1,
                  reference_answer='A', correct=True,
                  selection=dict(candidates=candidates, selected_frames=candidates, scores=[]),
                  answer=dict(pixel_shape=[16,3,384,384], visual_tokens=3360, prefill_tokens=3379,
                              text_input_tokens=20, raw_output='A.', parsed_answer='A'),
                  timings=dict(stage_sum_seconds=1, unattributed_seconds=0, end_to_end_seconds=1, blip_forward_seconds=0))
    results = tmp_path/'results/uniform'; results.mkdir(parents=True)
    (results/'q.json').write_text(json.dumps(record))
    attempts = tmp_path/'attempts/q'; attempts.mkdir(parents=True)
    (attempts/'uniform.01.json').write_text(json.dumps(dict(question_id='q', method='uniform', status='running_may_generate')))
    assert check_recovery(tmp_path, [row], 'p') == {('q','uniform')}
    with pytest.raises(ProtocolError, match='protocol'):
        check_recovery(tmp_path, [row], 'changed')
