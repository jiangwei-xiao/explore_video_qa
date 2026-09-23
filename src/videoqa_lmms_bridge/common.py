"""桥接身份、冻结名单与依赖校验。"""
import subprocess
from pathlib import Path
from videoqa_runtime.common import ROOT,read_json,sha256
from videoqa_full.state import durable,digest,utc,exclusive

LMMS_COMMIT='bb1ebe76e7a942386c25c4664f902e0e59e8a401'
WFS_COMMIT='a424fc4528ecbe57edc93413826a2f2b8bb2c203'
LMMS=ROOT/'third_party/lmms-eval-reference'
WFS=ROOT/'third_party/lmms-bridge-task-reference'
SOURCE=ROOT/'outputs/full_videoqa/rd_1_2__videomme2700__crossmodel_20260922__r01'
QUESTION_IDS=('264-2','184-3','004-3','102-2','395-2','305-3','409-3','844-1','872-2','815-2')
PROTOCOL='lmms-videomme-reference-v1'
CONDITIONS=('old','reference')
GENERATION=dict(max_new_tokens=16,temperature=0,top_p=1.0,num_beams=1,do_sample=False)
POST_PROMPT="\nAnswer with the option's letter from the given choices directly."


def manifest_rows():
    """接口：固定2700题，本阶段仅映射字段，不重新抽样。"""
    from videoqa_full.state import rows
    return rows()


def question_rows():
    """固定短4/中3/长3的10题，避免按答案或方法得失挑选。"""
    byid={r['question_id']:r for r in manifest_rows()}
    rows=[byid[q] for q in QUESTION_IDS]
    assert [r['stratum'] for r in rows]==['short']*4+['medium']*3+['long']*3
    return rows


def source_selection(qid,method='BASE-Uniform'):
    """接口：只返回历史源帧及身份；标准答案和历史预测不交给选帧模块。"""
    path=SOURCE/'selections'/method/(qid+'.json');spec=read_json(SOURCE/'protocol.json')
    if method=='RD-1.2':
        h=read_json(SOURCE/'llava/results/RD-1.2'/(qid+'.json'))['selection_sha256']
    else:h=spec['selection_exports'][str(path.relative_to(SOURCE))]
    assert sha256(path)==h
    return read_json(path),h


def framework_identity():
    """必须是指定提交、干净副本；任务参考文件单独保存，不套用全补丁。"""
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=LMMS,text=True).strip()==LMMS_COMMIT
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=LMMS,text=True).strip()
    return dict(commit=LMMS_COMMIT,model_sha256=sha256(LMMS/'lmms_eval/models/simple/llava_vid.py'),
        qwen_model_sha256=sha256(LMMS/'lmms_eval/models/simple/qwen2_5_vl.py'),
        instance_sha256=sha256(LMMS/'lmms_eval/api/instance.py'),task_reference_commit=WFS_COMMIT,
        task_sha256=sha256(WFS/'videomme.yaml'),task_utils_sha256=sha256(WFS/'utils.py'))


def task_doc(row):
    """框架计分字段与冻结原题映射；frame provider不接收此含答案对象。"""
    return dict(question_id=row['question_id'],videoID=row['video_id'],question=row['question'],
        options=row['options'],answer='ABCD'[row['answer_index']],duration=row['stratum'],
        domain=row['domain'],sub_category=row['original']['sub_category'],task_type=row['task_type'])


def load_task():
    """直接执行固定WFS任务函数，避免手抄另一个提示词或解析器冒充框架。"""
    import importlib.util
    spec=importlib.util.spec_from_file_location('videoqa_reference_videomme',WFS/'utils.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module
