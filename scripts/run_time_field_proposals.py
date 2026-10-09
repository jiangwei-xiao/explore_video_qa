"""显式时间字段先行提议：固定字面19题，先5题，问答/BLIP/视觉编码新增均0。"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys
import time

os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='')
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import rows,durable,exclusive,utc
from videoqa_audit.framework_contraction import BASE
from videoqa_audit.text_state_assets import ocr_identity,OCRProcess
from videoqa_audit.time_field_assets import extract_coarse_question
from videoqa_audit.explicit_time_fields import question_mode,extract_time_fields,propose_time_regions

LEGACY=ROOT/'outputs/methods/rd12_exp29_text_state_20261009_r3'
FONT=Path('/App/conda/envs/explore_video_qa/lib/python3.10/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf')
PILOT=['555-2','560-2','599-1','252-3','553-3']
# 这些源见证仅在提议保存后评价，不传入字段检测、语境支持或OCR。
EVALUATION={'555-2':119,'560-2':190,'599-1':21,'252-3':55,'553-3':511}
FILES=('scripts/run_time_field_proposals.py','src/videoqa_audit/explicit_time_fields.py',
       'src/videoqa_audit/time_field_assets.py','src/videoqa_audit/text_state_assets.py',
       'scripts/text_state_ocr_worker.py','src/videoqa_methods/density_selection.py')


def prepare(root):
    """接口：固定字面19题及完整粗池，冻结新OCR最大额度，不用正确性筛选总体。"""
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    with exclusive(root/'prepare.lock'):
        if (root/'protocol.json').exists(): return verify(root)
        data=[r for r in rows() if question_mode(r['question']) is not None]
        if len(data)!=19: raise ValueError('Literal probe population changed')
        old_spec=read_json(BASE/'protocol.json'); legacy_spec=read_json(LEGACY/'protocol.json'); ocr=ocr_identity()
        if ocr!=legacy_spec['ocr'] or sha256(ROOT/'scripts/text_state_ocr_worker.py')!=legacy_spec['files']['scripts/text_state_ocr_worker.py']:
            raise ValueError('OCR import backend identity differs')
        sources={}; imports={}; counts={}; videos={}
        # 步骤1：保存全部字面题及原分数身份，题面不进入OCR进程。
        for r in data:
            q=r['question_id']; p=BASE/'pools'/(q+'.json'); pool=read_json(p); check=BASE/'selection_checkpoints'/(q+'.json')
            if sha256(p)!=read_json(check)['files'][str(p.relative_to(BASE))]: raise ValueError('Coarse pool identity differs')
            coarse=pool['candidates'][:pool['initial_count']]; counts[q]=len(coarse); sources[str(p)]=sha256(p)
            vid=r['video_id']; videos[vid]=old_spec['video_hashes'][vid]
            for c in coarse:
                source=LEGACY/'frame_calls'/vid/(str(c['source_pts'])+'.json')
                if not source.exists(): continue
                call=read_json(source)
                if (call['status']!='completed' or call['protocol_sha256']!=sha256(LEGACY/'protocol.json')
                        or call['source_pts']!=c['source_pts'] or call['source_frame_index']!=c['source_frame_index']):
                    raise ValueError('Legacy OCR uncertain or another source')
                imports.setdefault(vid,{})[str(c['source_pts'])]=dict(path=str(source),sha256=sha256(source))
        # 步骤2：未识别/错字保持原始返回；没有新问答、BLIP或视觉编码阶段。
        spec=dict(reference_method='RD-1.2-exp07',kind='read_only_explicit_time_field_proposals',
            questions=[dict(question_id=r['question_id'],question=r['question'],mode=question_mode(r['question'])) for r in data],
            question_video_ids={r['question_id']:r['video_id'] for r in data},pilot_ids=PILOT,
            counts=counts,total_frame_positions=sum(counts.values()),maximum_new_ocr_frames=sum(counts.values())-sum(len(v) for v in imports.values()),
            imports=imports,legacy_protocol_sha256=sha256(LEGACY/'protocol.json'),pool_sources=sources,
            video_hashes=videos,ocr=ocr,files={p:sha256(ROOT/p) for p in FILES},font_sha256=sha256(FONT),
            context_radius_seconds=4,audit_top_positions=4,max_synthetic_ocr_frames=5,new_qa_limit=0,
            pilot_gate='At least2of3 known new source carriers among first4 unique proposals and both correct controls reachable; then inventory19 only',
            no_gold_options_or_manual_facts_in_proposal=True,utc=utc())
        for name in FILES:
            out=root/'code_snapshot'/name; out.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(ROOT/name,out)
        durable(root/'protocol.json',spec); return spec


def verify(root):
    """接口：恢复固定输入/依赖/代码，语法验证或观察超时不能授权重复模型前向。"""
    root=Path(root); spec=read_json(root/'protocol.json')
    if spec['new_qa_limit']!=0 or spec['pilot_ids']!=PILOT: raise ValueError('Probe scope changed')
    if ocr_identity()!=spec['ocr'] or sha256(FONT)!=spec['font_sha256']: raise ValueError('OCR environment/font changed')
    if sha256(LEGACY/'protocol.json')!=spec['legacy_protocol_sha256']: raise ValueError('Legacy OCR identity changed')
    for p,h in spec['files'].items():
        if sha256(ROOT/p)!=h: raise ValueError('Frozen probe code changed')
    for p,h in spec['pool_sources'].items():
        if sha256(p)!=h: raise ValueError('Frozen coarse asset changed')
    return spec


def synthetic(root):
    """接口：3种可见时刻、空白和订阅文字合成验收；最多5个OCR帧，非数据集QA。"""
    from PIL import Image,ImageDraw,ImageFont
    import numpy as np
    root=Path(root); verify(root); engine=None
    labels=[('am','10:00 AM'),('pm','7:30pm'),('clock','09:24'),('blank',''),('promo','PLEASE SUBSCRIBE')]
    try:
        for name,label in labels:
            out=root/'synthetic'/(name+'.json')
            if out.exists():
                if read_json(out)['status']!='completed': raise ValueError('Synthetic OCR uncertain; no retry')
                continue
            image=Image.new('RGB',(800,240),'black'); draw=ImageDraw.Draw(image)
            if label: draw.text((45,70),label,font=ImageFont.truetype(str(FONT),64),fill='white')
            rgb=np.asarray(image).copy(); image.close(); durable(out,dict(status='started'))
            if engine is None: engine=OCRProcess(root/'logs/synthetic.log')
            result=engine.infer(rgb)
            durable(out,dict(status='completed',ocr=result,fields=extract_time_fields(result['regions'],'clock')))
        if read_json(root/'synthetic/blank.json')['fields'] or read_json(root/'synthetic/promo.json')['fields']:
            raise ValueError('Synthetic false time field')
        found=sum(bool(read_json(root/'synthetic'/(n+'.json'))['fields']) for n in ('am','pm','clock'))
        if found<2: raise ValueError('Synthetic numerical field recognition insufficient')
        durable(root/'synthetic_acceptance.json',dict(status='passed',recognized=found,total_time_patterns=3,
            scope='Transport and visible format recognition, not semantic correctness of real fields'))
    finally:
        if engine is not None: engine.close()


def extract_group(job):
    """接口：同视频题顺序处理，共享完成源缓存，不并发争抢同一源前向。"""
    root,ids=job; return [extract_coarse_question((root,q)) for q in ids]


def extract(root,population):
    """接口：有界CPU后台扫描全部粗候选，缓存与新计算成本分开。"""
    root=Path(root); spec=verify(root)
    if read_json(root/'synthetic_acceptance.json')['status']!='passed': raise ValueError('Synthetic gate missing')
    ids=PILOT if population=='pilot' else [r['question_id'] for r in spec['questions']]
    grouped={}
    for q in ids: grouped.setdefault(spec['question_video_ids'][q],[]).append(q)
    jobs=sorted([(str(root),qs) for qs in grouped.values()],key=lambda v:-max(spec['counts'][q] for q in v[1]))
    with exclusive(root/(population+'_extract.lock')):
        started=time.perf_counter(); done=[]
        with ProcessPoolExecutor(max_workers=min(8,len(jobs)),mp_context=mp.get_context('spawn')) as executor:
            for group in executor.map(extract_group,jobs):
                done+=group; durable(root/'status.json',dict(status='extracting',population=population,banks_completed=len(done),banks_total=len(ids),
                    completed_source_positions=sum(x['frames'] for x in done),pid=os.getpid(),sid=os.getsid(0),utc=utc()))
        calls=[read_json(p) for p in (root/'frame_calls').rglob('*.json')]
        if any(c['status']!='completed' for c in calls): raise ValueError('OCR bank includes unknown forward')
        fresh=[c for c in calls if not c['reused']]
        if len(fresh)>spec['maximum_new_ocr_frames']: raise ValueError('Frozen OCR budget exceeded')
        durable(root/(population+'_extract_summary.json'),dict(status='completed',banks=len(done),new_frame_calls=len(fresh),
            reused_frame_calls=len(calls)-len(fresh),detection_forwards=sum(c['ocr']['detection_forwards'] for c in fresh),
            recognition_batches=sum(len(c['ocr']['recognition_batches']) for c in fresh),seconds=time.perf_counter()-started,new_qa_calls=0))


def propose(root,population):
    """接口：不读答案生成字段提议，再单独以已知源见证评价可达性。"""
    root=Path(root); spec=verify(root); bank={r['question_id']:r for r in spec['questions']}
    ids=PILOT if population=='pilot' else list(bank); exports={}
    # 步骤1：每题独立消费原始OCR和既有分数，提议函数完全不接收人工见证。
    for q in ids:
        pool=read_json(BASE/'pools'/(q+'.json')); coarse=pool['candidates'][:pool['initial_count']]
        assets=read_json(root/'ocr_banks'/(q+'.json')); values=[]
        for c in coarse:
            path=root/'frame_calls'/spec['question_video_ids'][q]/(str(c['source_pts'])+'.json')
            if sha256(path)!=assets['calls'][str(path)]: raise ValueError('Actual OCR return changed')
            values.append(extract_time_fields(read_json(path)['ocr']['regions'],bank[q]['mode']))
        result=propose_time_regions(coarse,pool['initial_scores'],values,pool['video'])
        out=root/'proposals'/(q+'.json')
        value=dict(question_id=q,mode=bank[q]['mode'],fields=values,**result,
            sources={str(BASE/'pools'/(q+'.json')):sha256(BASE/'pools'/(q+'.json')),str(root/'ocr_banks'/(q+'.json')):sha256(root/'ocr_banks'/(q+'.json'))},
            protocol_sha256=sha256(root/'protocol.json'),new_qa_calls=0)
        if out.exists() and read_json(out)!=value: raise ValueError('Proposal changed on recovery')
        if not out.exists(): durable(out,value)
        exports[str(out)]=sha256(out)
    # 步骤2：评价使用独立目录；这里只判断输入提议，不把前4可达称作事件绑定成功。
    checks=[]
    for q,index in EVALUATION.items():
        if q not in ids: continue
        p=read_json(root/'proposals'/(q+'.json'))
        matches=[v for v in p['proposals'] if any(index in r['members'] for r in v['regions'])]
        best=min((v['rank'] for v in matches),default=None)
        checks.append(dict(question_id=q,coarse_witness_index=index,recognized=bool(p['fields'][index]),best_carrier_rank=best,
            first4_reachable=best is not None and best<=4,visible_fields=p['fields'][index]))
    target=sum(c['first4_reachable'] for c in checks if c['question_id'] in PILOT[:3])
    controls=sum(c['first4_reachable'] for c in checks if c['question_id'] in PILOT[3:])
    passed=target>=2 and controls==2
    summary=dict(status='completed',population=population,files=exports,evaluation=checks,
        main_carriers_in_first4=target,correct_controls_in_first4=controls,pilot_inventory_gate=passed,
        additional_language_or_QA_calls=0,selection_writes=0,
        scope='Listed-source carrier proposal audit, not event alignment or algorithm performance')
    durable(root/(population+'_proposal_summary.json'),summary); return summary


def main():
    """接口：后台先合成、5题提议门槛，再决定是否只扩展19题库存；不会自动问答。"""
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--stage',choices=('prepare','synthetic','pilot','all','verify'),default='all');a=p.parse_args()
    if not a.run_id.replace('_','').isalnum():p.error('Invalid run ID')
    root=ROOT/'outputs/analysis'/a.run_id
    try:
        if a.stage=='prepare': prepare(root)
        elif a.stage=='verify': verify(root)
        elif a.stage=='synthetic': synthetic(root)
        elif a.stage=='pilot': extract(root,'pilot');propose(root,'pilot')
        else:
            with exclusive(root/'pipeline.lock'):
                prepare(root);synthetic(root);extract(root,'pilot');result=propose(root,'pilot')
                if result['pilot_inventory_gate']:
                    extract(root,'all');propose(root,'all')
                durable(root/'status.json',dict(status='completed',pilot_inventory_gate=result['pilot_inventory_gate'],
                    whole19_executed=result['pilot_inventory_gate'],new_qa_calls=0,pid=os.getpid(),sid=os.getsid(0),utc=utc()))
    except BaseException as exc:
        durable(root/'status.json',dict(status='paused',error=repr(exc),pid=os.getpid(),sid=os.getsid(0),utc=utc()));raise


if __name__=='__main__':main()
