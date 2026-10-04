"""固定官方核心默认配置在现有13题粗池上的离线参考，不声称复现论文成绩。"""
import os
os.environ.update(OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
import importlib.util
from dataclasses import asdict
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from videoqa_runtime.common import read_json, sha256
from videoqa_full.state import durable, exclusive, utc
from videoqa_audit.query_input_probe import BASE


def main():
    """接口：只换为固定公开核心选择器，使用原BLIP粗评分/CLS，最多32帧，无问答。"""
    run = ROOT/'outputs/analysis/wfs_core_reference13_20261004_r1'
    code = ROOT/'third_party/WFS-SB-reference/core.py'
    panel = ROOT/'outputs/analysis/joint_temporal_support13_20261004_r1/analysis.json'
    with exclusive(run/'run.lock'):
        # 步骤1：验证已冻结官方核心字节，不重写或优化参考算法。
        expected = read_json(ROOT/'configs/method_v1.json')['wfs_reference_sha256']
        if sha256(code) != expected: raise ValueError('Reference core changed')
        spec = importlib.util.spec_from_file_location('fixed_wfs_core_reference', code)
        module = importlib.util.module_from_spec(spec);sys.modules[spec.name] = module;spec.loader.exec_module(module)
        cfg = module.WFSConfig();selector = module.WFS(cfg)
        import scipy, sklearn, pywt
        ids = [r['question_id'] for r in read_json(panel)['records']]
        sources = {str(p):sha256(p) for p in [Path(__file__),code,panel]}
        for q in ids:
            for folder, suffix in [('pools','.json'),('features','.npy')]:
                p=BASE/folder/(q+suffix);sources[str(p)]=sha256(p)
        protocol = dict(sources=sources, question_ids=ids, config=asdict(cfg), frame_budget=32,
            environment=dict(python=sys.executable,numpy=np.__version__,scipy=scipy.__version__,
                             sklearn=sklearn.__version__,pywavelets=pywt.__version__),
            reference_commit='a424fc4528ecbe57edc93413826a2f2b8bb2c203',
            scorer='frozen BLIP-ITM-base-coco original-question probabilities',
            features='frozen pure visual normalized CLS, coarse prefix only', new_model_calls=0,
            interpretation='公开核心默认值与本地冻结资产的结构对照；非论文主配置或成绩复现')
        pp=run/'protocol.json'
        if pp.exists():
            if read_json(pp)!=protocol:raise ValueError('Reference protocol changed')
        else:durable(pp,protocol)
        # 步骤2：直接调用官方函数，保留原短池、无边界回退、预算和MMR同分行为。
        for q in ids:
            dest=run/'selections'/(q+'.json')
            if dest.exists():
                if read_json(dest)['protocol_sha256']!=sha256(pp):raise ValueError('Selection provenance changed')
                continue
            pool=read_json(BASE/'pools'/(q+'.json'));n=pool['initial_count']
            scores=np.asarray(pool['initial_scores'],dtype=np.float64)
            features=np.load(BASE/'features'/(q+'.npy'),allow_pickle=False)[:n]
            level=module.compute_dwt_level(n)
            distance=module.compute_min_peak_distance(n)
            selected=list(map(int,selector.select_keyframes(scores,32,level,distance,features)))
            if len(selected)!=min(32,n) or len(set(selected))!=len(selected) or selected!=sorted(selected):
                raise ValueError('Reference selection violates source budget')
            durable(dest,dict(question_id=q,protocol_sha256=sha256(pp),selected_indices=selected,
                dwt_level=level,min_peak_distance=distance,initial_count=n,
                selection=dict(video=pool['video'],selected_frames=[pool['candidates'][i] for i in selected])))
            durable(run/'status.json',dict(status='running',question_id=q,pid=os.getpid(),sid=os.getsid(0),utc=utc()))
            print(q,'selected',len(selected),flush=True)
        # 步骤3：只输出可追溯选择，绝不沿用旧答案给新参考计算准确率。
        durable(run/'summary.json',dict(status='completed',questions=len(ids),new_model_calls=0,new_answers=0,
            files={str(run/'selections'/(q+'.json')):sha256(run/'selections'/(q+'.json')) for q in ids}))
        durable(run/'status.json',dict(status='completed',pid=os.getpid(),sid=os.getsid(0),utc=utc()))


if __name__=='__main__':main()
