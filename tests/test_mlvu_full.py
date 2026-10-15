"""MLVU独立任务适配、宏平均、长PTS和真实输入共享的零模型回归。"""
import json
from pathlib import Path
import numpy as np
import pytest
from videoqa_mlvu.data import TASKS, context, task_doc
from videoqa_mlvu.backends import metric, observer_module
from videoqa_mlvu.runtime import source_order
from videoqa_mlvu.report import score_table, compare
from videoqa_budget32.frames import native_frames


def row():
    return dict(question='Which object?', qa_question='Which object?\n(A) Fork\n(B) Spoon\n(C) Cup\n(D) Book\n',
                options=['Fork','Spoon','Cup','Book'], answer_index=0, task_type='ego')


def test_native_prompt_and_answer_separation():
    """问答原生后缀和选项只出现一次，改变标准答案不应改变模型文字。"""
    a = row(); before = context(a); a['answer_index'] = 3
    assert context(a) == before
    assert before.endswith('\nOnly give the best option.\nBest option: (')
    assert before.count('(A) Fork') == 1
    assert a['question'] == 'Which object?' and a['question'] not in a['options']


@pytest.mark.parametrize('text,expected', [('A)', 'A'), ('B', 'B'), (' C) ', 'C'), ('unclear', 'unclear')])
def test_fixed_native_parser_not_videomme_parser(text, expected):
    assert metric(row(), [text])['pred_answer'] == expected


@pytest.mark.parametrize('profile', ['qwen32','llava8'])
def test_observer_task_binding_is_private(profile):
    """私有观测实例不污染旧框架模块任务绑定，也不改原源文件。"""
    from videoqa_runtime.common import ROOT, sha256
    path = ROOT / ('src/videoqa_budget32/observe.py' if profile == 'qwen32' else 'src/videoqa_generalization/llava_observe.py')
    before = sha256(path); module = observer_module(profile)
    assert sha256(path) == before
    assert module.load_task().videomme_doc_to_text(task_doc(row()), {}) == context(row())
    assert module.load_task().videomme_process_results(task_doc(row()), ['A)'])['videomme_perception_score'] == metric(row(), ['A)'])


def test_long_pts_and_json_roundtrip():
    """超32位PTS与九小时非均匀时间轴保持真实时间；短源池不复制凑预算。"""
    # 步骤1：原点超过有符号32位，故意让中间帧远离平均FPS推算的位置。
    video = dict(start_pts=2**31+100, time_base='1', duration_fraction='32551', duration_seconds=32551.)
    pts = np.asarray([2**31+100, 2**31+200, 2**31+32650], dtype=np.int64)
    frames = native_frames(pts, video, 3); selection = dict(video=video, selected_frames=frames)
    assert source_order(selection) == json.loads(json.dumps(source_order(selection)))
    assert frames[-1]['source_pts'] > 2**31 and frames[-1]['timestamp_seconds'] == 32550.
    # 步骤2：逐帧检查真实时间和编号，而不是只核查首尾；预算上限不改变源身份。
    assert [f['timestamp_seconds'] for f in frames] == [0., 100., 32550.]
    assert [f['source_frame_index'] for f in frames] == [0, 1, 2]
    assert native_frames(pts, video, 8) == frames


def test_macro_not_overall_accuracy():
    data = []; bank = {}
    for t in TASKS:
        for i in range(10 if t == 'plotQA' else 1):
            q = t+str(i); data.append(dict(question_id=q, task_type=t)); bank[q] = dict(correct=t == 'plotQA')
    stats = score_table(data, bank)
    assert stats['accuracy'] == 10/16
    assert stats['m_avg_percent'] == pytest.approx(100/7)


def test_video_cluster_preserves_cross_task_correlation():
    data = []; left = {}; right = {}
    for video in ('one','two'):
        for t in TASKS:
            q = video+t; data.append(dict(question_id=q, task_type=t, video_id=video))
            left[q] = dict(correct=video == 'one'); right[q] = dict(correct=video == 'two')
    result = compare(data, left, right, repetitions=1000)
    assert result['clusters'] == 2 and result['m_avg_difference_pp'] == 0
    assert result['m_avg_video_cluster_bootstrap_95'] == [-100.,100.]


def test_actual_official_manifest_mapping_if_prepared():
    from videoqa_runtime.common import ROOT, read_json
    path = ROOT / 'outputs/full_videoqa/mlvu_dev__S1__llava8_qwen32__four_methods__20261014_r1/manifest.json'
    if not path.exists(): pytest.skip('MLVU data preparation not available')
    data = read_json(path)['rows']
    assert len(data) == len({r['official_question_id'] for r in data}) == 2174
    assert {r['official_question_id'] for r in data if r['ambiguous_correct_options']} == {'Q421','Q1448','Q1500','Q1606','Q1814'}


def test_coarse_shared_decode_scores_only_original_questions(tmp_path, monkeypatch):
    """共享解码不共享题目分数，且不会把QA中的选项喂给BLIP。"""
    from videoqa_mlvu import preparation as p
    from videoqa_full.state import durable, save_array
    from PIL import Image
    video = dict(path='fake.mp4', time_base='1/1000', start_pts=0, duration_seconds=2., duration_fraction='2', candidate_fps=1, phase_seconds=.25)
    path = tmp_path / 'indices/v.npy'; save_array(path, np.asarray([0,300,1300], dtype=np.int64))
    durable(tmp_path / 'indices/v.json', dict(video=video, pts_path=str(path)))
    durable(tmp_path / 'protocol.json', dict(testing=True))
    class Frame:
        def to_image(self): return Image.new('RGB', (8,8), 'blue')
    def iterator(path, metadata):
        metadata.update({k:v for k,v in video.items() if k != 'duration_fraction'})
        for i,(source,pts,target) in enumerate([(1,300,.25),(2,1300,1.25)]):
            yield dict(candidate_index=i,source_frame_index=source,source_pts=pts,timestamp_seconds=pts/1000,requested_seconds=target),Frame()
    questions = []
    class Scorer:
        def tokenize(self, text): questions.append(text); return dict(input_ids=np.zeros((1,3), dtype=int))
        def score_batch_features(self, text, images):
            return [.5]*len(images),np.tile(np.asarray([1.,0.],dtype=np.float32),(len(images),1)),0.,0.
    monkeypatch.setattr(p, 'iter_candidate_frames', iterator); monkeypatch.setattr(p, '_SCORER', Scorer())
    p.coarse_video(tmp_path, [dict(question_id='q1',video_id='v',question='original one'),
                              dict(question_id='q2',video_id='v',question='original two')])
    assert questions == ['original one','original two']
    assert p.completed(tmp_path,'coarse','q1') and p.completed(tmp_path,'coarse','q2')
