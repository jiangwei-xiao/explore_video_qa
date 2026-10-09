"""exp29冻结协议、8题OCR实际竞争与有条件双配置开发；不改父算法。"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import os,time,shutil
import numpy as np
from videoqa_runtime.common import ROOT,read_json,sha256
from videoqa_full.state import rows,durable,exclusive,utc
from videoqa_audit.framework_contraction import BASE,locations
from videoqa_audit.region_scope_experiments import DEV,FACTS,profile_budget
from videoqa_methods.text_state_redundancy import select_text_state
from videoqa_audit.text_state_assets import ocr_identity,extract_question,OCRProcess
from videoqa_budget32.frames import validate_frames

METHOD='RD-1.2-exp29'
FONT=Path('/App/conda/envs/explore_video_qa/lib/python3.10/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf')
PILOT=('007-2','022-1','035-2','584-3','191-3','538-1','103-1','383-2')
TARGETS={'007-2':[27],'022-1':[25],'035-2':[55,56,59],'584-3':[582],'191-3':[6]}
FILES=('src/videoqa_methods/text_state_redundancy.py','src/videoqa_audit/text_state_assets.py',
       'src/videoqa_audit/text_state_experiments.py','scripts/text_state_ocr_worker.py',
       'scripts/run_text_state_experiments.py','src/videoqa_methods/weighted_seed_support.py',
       'src/videoqa_methods/density_selection.py','configs/retrieval_density_v1.json')


def development_ids():
    """接口：仅使用原冻结200题清单，不重抽样或以新输出替换样本。"""
    ids=[r['question_id'] for r in read_json(DEV)['rows']]
    if len(ids)!=200 or len(set(ids))!=200:raise ValueError('Development population differs')
    return ids


def prepare(root):
    """接口：一次冻结试点及开发的候选/OCR身份、400问答上限和退出条件。"""
    root=Path(root);root.mkdir(parents=True,exist_ok=True);bank={r['question_id']:r for r in rows()}
    ids=list(dict.fromkeys(list(PILOT)+development_ids()));sources={}
    frames={}
    for q in ids:
        p=BASE/'pools'/(q+'.json');pool=read_json(p);sources[str(p)]=sha256(p)
        frames[q]=len(pool['candidates'])
    spec=dict(method=METHOD,parent='RD-1.2-exp07',pilot_ids=list(PILOT),development_ids=development_ids(),
        question_video_ids={q:bank[q]['video_id'] for q in ids},allowed_ocr_question_ids=ids,
        maximum_ocr_frame_positions=sum(frames.values()),frames_per_bank=frames,ocr=ocr_identity(),
        files={n:sha256(ROOT/n) for n in FILES},pool_sources=sources,
        development_sha256=sha256(DEV),facts_sha256=sha256(FACTS),text_score=0.5,
        synthetic_font=dict(path=str(FONT),sha256=sha256(FONT)),
        rule='Only candidates with recognized text attenuate visual redundancy by OCR token Jaccard; fixed parent quotas',
        real_qa_limit_per_profile=200,real_qa_limit_total=400,
        pilot_gate='At least two distinct target videos improve listed-state prefix/input; no confirmed unique evidence loss in protected controls',
        full_gate='Both development scores >= exp07 and (Qwen>=140 or LLaVA>=128); full requires stage review',
        no_query_or_classifier=True,new_blip_calls=0,new_visual_cls_calls=0)
    with exclusive(root/'prepare.lock'):
        p=root/'protocol.json'
        if p.exists():
            if read_json(p)!=spec:raise ValueError('Frozen text-state protocol changed')
        else:
            durable(p,spec)
            for n in FILES:
                dest=root/'code_snapshot'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,dest)
        return spec


def verify(root):
    """接口：辅助环境、方法、资产或清单变化时暂停，未完成OCR不得重发。"""
    root=Path(root);spec=read_json(root/'protocol.json')
    if ocr_identity()!=spec['ocr']:raise ValueError('Frozen OCR environment changed')
    for n,h in spec['files'].items():
        if sha256(ROOT/n)!=h:raise ValueError('Frozen method code changed')
    for n,h in spec['pool_sources'].items():
        if sha256(n)!=h:raise ValueError('Frozen parent pool changed')
    return spec


def synthetic(root):
    """接口：3张合成图片验收默认检测/识别与空文字回退，无开发题答案。"""
    from PIL import Image,ImageDraw,ImageFont
    from videoqa_methods.text_state_redundancy import normalize_text_state
    root=Path(root);verify(root);process=None
    try:
        for name,text in [('blank',''),('alpha','STATE ALPHA 12'),('beta','STATE BETA 13')]:
            path=root/'synthetic'/(name+'.json')
            if path.exists():
                if read_json(path)['status']!='completed':raise ValueError('Uncertain synthetic OCR')
                continue
            im=Image.new('RGB',(640,160),'white');draw=ImageDraw.Draw(im)
            if text:draw.text((20,50),text,font=ImageFont.truetype(str(FONT),42),fill='black')
            rgb=np.asarray(im).copy();im.close();durable(path,dict(status='started'))
            if process is None:process=OCRProcess(root/'logs/synthetic_ocr.log')
            result=process.infer(rgb);tokens=normalize_text_state(result['regions'])
            durable(path,dict(status='completed',ocr=result,tokens=tokens))
        # 合成只验证空白与不同可见文字状态可区分，不要求CTC逐词空格完全复现。
        # 原始返回必须先落盘；真实标题/编号的识别正确性在独立试点画面中审核。
        blank=read_json(root/'synthetic/blank.json')['tokens']
        alpha=read_json(root/'synthetic/alpha.json')['tokens'];beta=read_json(root/'synthetic/beta.json')['tokens']
        if blank or not alpha or not beta or alpha==beta:raise ValueError('Synthetic OCR state distinction failed')
        durable(root/'synthetic_acceptance.json',dict(status='passed',scope='Transport and distinct text states; not verbatim OCR accuracy',
            alpha_tokens=alpha,beta_tokens=beta,blank_tokens=blank))
    finally:
        if process is not None:process.close()


def extract_video_group(job):
    """接口：同视频问题顺序提取，避免共享源缓存被两个进程同时请求。"""
    root,questions=job
    return [extract_question((root,q)) for q in questions]


def extract(root,population,workers=8):
    """接口：有界CPU并行提取，初次与缓存复用成本分开，不生成QA。"""
    root=Path(root);spec=verify(root);ids=spec['pilot_ids'] if population=='pilot' else spec['development_ids']
    grouped={}
    for q in ids:grouped.setdefault(spec['question_video_ids'][q],[]).append(q)
    started=time.perf_counter();results=[]
    with exclusive(root/(population+'_extract.lock')):
        # 步骤1：同源图像最多一次已确认前向，不通过删除未知账本实现恢复。
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for batch in executor.map(extract_video_group,[(str(root),qs) for qs in grouped.values()]):
                results+=batch;durable(root/(population+'_extract_status.json'),dict(completed=len(results),total=len(ids),
                    pid=os.getpid(),sid=os.getsid(0),seconds=time.perf_counter()-started,utc=utc()))
        calls=[read_json(p) for p in (root/'frame_calls').rglob('*.json')]
        value=dict(status='completed',banks=len(results),source_frame_calls=len(calls),
                   detection_forwards=sum(x['ocr']['detection_forwards'] for x in calls),
                   recognition_batches=sum(len(x['ocr']['recognition_batches']) for x in calls),
                   new_dataset_answers=0,seconds=time.perf_counter()-started)
        durable(root/(population+'_extract_summary.json'),value);return value


def select_question(root,q,profile):
    """接口：同一个OCR源银行映射K32/K8，固定exp07配额和全部上游身份。"""
    root=Path(root);k=profile_budget(profile);pp,fp,tp,sp=locations(q,k)
    ocr_path=root/'ocr_banks'/(q+'.json');ocr=read_json(ocr_path);pool=read_json(pp);global_f=np.load(fp,allow_pickle=False)
    mapping=list(range(len(global_f))) if k==32 else pool['original_candidate_indices']
    if ocr['pool_sha256']!=sha256(BASE/'pools'/(q+'.json')):raise ValueError('OCR bank identity changed')
    keys=[[f['source_frame_index'],f['source_pts']] for f in pool['candidates']]
    if keys!=[ocr['source_keys'][i] for i in mapping]:raise ValueError('OCR/feature/source mapping differs')
    states=[ocr['states'][i] for i in mapping];f=global_f if k==32 else global_f[mapping]
    parent=read_json(tp);parent=parent['exp07'] if k==8 else parent
    selected,trace=select_text_state(pool,f,parent,states,k)
    selection=dict(video=pool['video'],selected_frames=[pool['candidates'][i] for i in selected]);validate_frames(selection,k)
    control_states=[[] for _ in states];control,_=select_text_state(pool,f,parent,control_states,k)
    expected=read_json(sp)['selection'];key=lambda s:[(f['source_frame_index'],f['source_pts']) for f in s['selected_frames']]
    if key(dict(video=pool['video'],selected_frames=[pool['candidates'][i] for i in control]))!=key(expected):raise ValueError('No-text control fails original selection')
    trp=root/'traces'/profile/(q+'.json');durable(trp,trace)
    value=dict(question_id=q,profile=profile,method=METHOD,selection=selection,selected_indices=selected,
        actual_frame_count=len(selected),unchanged_selected_sources=key(selection)==key(expected),parent_quotas_fixed=True,
        states=states,source_pool=str(pp),source_features=str(fp),trace_file=str(trp.relative_to(root)),trace_sha256=sha256(trp),
        sources={str(p):sha256(p) for p in (pp,fp,tp,sp,ocr_path)},no_text_control_exact=True,new_qa_calls=0)
    path=root/'offline'/profile/(q+'.json')
    if path.exists():
        old=read_json(path)
        if old!=value:raise ValueError('Selection recovery changed')
    else:durable(path,value)
    return value


def select_population(root,population):
    """接口：选择独立于OCR和QA，恢复核验同一输入，不根据答案改变规则。"""
    root=Path(root);spec=verify(root);ids=spec['pilot_ids'] if population=='pilot' else spec['development_ids'];values=[]
    for profile in ('qwen32','llava8'):
        for q in ids:values.append(select_question(root,q,profile))
    value=dict(status='passed',questions=len(ids),selections=len(values),
        files={str(root/'offline'/x['profile']/(x['question_id']+'.json')):sha256(root/'offline'/x['profile']/(x['question_id']+'.json')) for x in values})
    durable(root/(population+'_selection_summary.json'),value);return value
