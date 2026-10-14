"""评分器对照隔离恢复：只规范JSON源身份容器，不改模型/帧/提示和冻结文件。"""
import json
from pathlib import Path
from videoqa_runtime.common import ROOT,read_json,sha256
from videoqa_full.state import durable,digest,utc
from videoqa_audit.region_scope_runtime import source_order as original_source_order


def canonical_source_order(selection):
    """接口：元组仅转成JSON数组，语义摘要必须保持不变；源编号/PTS/顺序一律不改。"""
    # 步骤1：复用冻结源身份定义，只消除Python容器和JSON落盘容器的类型差别。
    source=original_source_order(selection)
    normalized=json.loads(json.dumps(source,allow_nan=False))
    if digest(source)!=digest(normalized):raise ValueError('Canonicalization changed source semantics')
    return normalized


def install_recovery(root):
    """接口：额外执行代码单独绑定指纹，旧protocol/快照/选择/调用文件保持字节不变。"""
    from videoqa_scorer_comparison import runtime
    root=Path(root);path=root/'operational_recovery/source_identity_r2.json'
    code={n:sha256(ROOT/n) for n in ['src/videoqa_audit/scorer_comparison_recovery.py','scripts/resume_scorer_comparison.py']}
    frozen={str(root/'preflight.json'):sha256(root/'preflight.json')}
    for profile in ('qwen32','llava8'):
        p=root/'qa'/profile/'protocol.json'
        if p.exists():frozen[str(p)]=sha256(p)
    if path.exists():
        old=read_json(path)
        if old['code_sha256']!=code:raise ValueError('Operational recovery code changed')
        for p,h in old['frozen_sources'].items():
            if sha256(p)!=h:raise ValueError('Frozen recovery input changed')
    else:
        durable(path,dict(status='approved_execution_fix',issue='tuple versus JSON list source identity',
            code_sha256=code,frozen_sources=frozen,model_or_selection_change=False,new_qa_budget_added=0,utc=utc()))
        for n in code:
            import shutil
            dest=root/'operational_recovery/code_snapshot'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,dest)
    # 步骤2：只替换新运行层内存中的表示函数；不编辑原运行层或历史帮助函数。
    runtime.source_order=canonical_source_order


def recovery_worker(root,stage,gpu,jobs,events,stop):
    """接口：spawn子进程各自安装同一表示修复，不依赖父进程的内存补丁继承。"""
    install_recovery(root)
    from videoqa_scorer_comparison.coordinator import worker
    return worker(root,stage,gpu,jobs,events,stop)


def validate_imports(root):
    """接口：已导入的74个Qwen结果实际像素/Token/源身份逐项验收，模型调用0。"""
    from videoqa_scorer_comparison import runtime
    from videoqa_full.state import rows
    root=Path(root);run=root/'qa/qwen32';spec=runtime.verify(run);n=0
    for row in rows():
        for method in runtime.METHODS:
            p=run/'results'/method/(row['question_id']+'.json')
            if p.exists():runtime.validate_result(run,row,method,spec);n+=1
    durable(root/'operational_recovery/import_acceptance.json',dict(status='passed',results=n,
        actual_input_validation=True,new_model_calls=0,utc=utc()));return n
