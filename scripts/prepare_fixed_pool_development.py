"""冻结开发200题M16/K32旧池诊断输入：不评分、不补查、不读取标准答案选帧。"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable,rows,exclusive,utc
from videoqa_budget32.frames import FrameProvider,validate_frames
from videoqa_budget32.selection import SOURCE

FULL32=ROOT/'outputs/full_videoqa/uniform_topk_rd12__qwen25vl__lmms_ref_v1__videomme2700__k32_20260924__r01'
MANIFEST=ROOT/'data/manifests/video_mme_development_200_20260920.json'

def prepare_one(job):
    """接口：从已经冻结的离线诊断读取精确帧，只补齐RGB身份，供统一后端复用。"""
    directory,row=job;q=row['question_id'];dest=directory/(q+'.json')
    dp=FULL32/'diagnostics'/(q+'.json');diag=read_json(dp)
    pp=SOURCE/'llava/pools'/(q+'.json');pool=read_json(pp)
    sources={str(dp):sha256(dp),str(pp):sha256(pp)}
    for p,h in diag['source_hashes'].items():assert sha256(p)==h
    sources.update(diag['source_hashes'])
    # 步骤1：只接受原始M16/K32诊断；短池按实际数量，不补帧凑数。
    assert diag['configuration']=='M16/K32 fixed old pool' and diag['new_model_calls']==0
    selection=dict(video=pool['video'],selected_frames=diag['selected_frames'])
    validate_frames(selection)
    if dest.exists():
        saved=read_json(dest);assert saved['sources']==sources and saved['selection']==selection
        return q,sha256(dest)
    # 步骤2：精确PTS解码，像素身份将在正式模型调用前再次验证。
    images,hashes=FrameProvider(selection).decode()
    for im in images:im.close()
    durable(dest,dict(question_id=q,video_id=row['video_id'],method='RD-1.2',configuration='M16/K32 fixed old pool',
        selection=selection,rgb_sha256=hashes,sources=sources))
    return q,sha256(dest)

def main(output):
    """接口：顺序以冻结开发清单为准；准备阶段只读原资产，允许指纹一致恢复。"""
    directory=Path(output);directory.mkdir(parents=True,exist_ok=True)
    with exclusive(directory/'prepare.lock'):
        dev=read_json(MANIFEST)['rows'];full={r['question_id']:r for r in rows()}
        data=[full[r['question_id']] for r in dev]
        assert len(data)==len({r['question_id'] for r in data})==200
        for a,b in zip(dev,data):
            for key in ('question_id','video_id','question','options','answer_index'):assert a[key]==b[key]
        manifest=directory/'selection_manifest.json'
        if manifest.exists():
            old=read_json(manifest);assert old['manifest_sha256']==sha256(MANIFEST)
            assert old['preparer_sha256']==sha256(__file__)
            for q,h in old['exports'].items():
                p=directory/(q+'.json');assert sha256(p)==h
                for source,expected in read_json(p)['sources'].items():assert sha256(source)==expected
            print('Frozen inputs already complete; no model calls',flush=True);return
        # 步骤3：CPU并行解码；不加载问答/BLIP模型，元数据最后原子提交。
        with ThreadPoolExecutor(max_workers=8) as ex: exports=dict(ex.map(prepare_one,[(directory,r) for r in data]))
        durable(manifest,dict(status='completed',method='RD-1.2',configuration='M16/K32 fixed old pool',diagnostic=True,
            question_ids=[r['question_id'] for r in data],question_count=200,frame_budget=32,
            dataset_manifest=str(MANIFEST),manifest_sha256=sha256(MANIFEST),preparer_sha256=sha256(__file__),
            parent_run=str(FULL32),parent_protocol_sha256=sha256(FULL32/'protocol.json'),exports=exports,new_model_calls=0,utc=utc()))
        print('Frozen 200 inputs; new model calls 0',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);main(parser.parse_args().output)
