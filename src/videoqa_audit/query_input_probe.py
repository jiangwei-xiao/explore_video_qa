"""固定案例的定位文本对照：复核原题，再比较全部选项输入；不生成问答或修改冻结方法。"""
import hashlib
import os
from pathlib import Path
import subprocess
import time
import traceback
import numpy as np
from videoqa_runtime.common import ROOT,read_json,sha256,offline_environment
from videoqa_full.state import durable,exclusive,utc,save_array
from videoqa_runtime.baseline_selection import iter_candidate_frames,ProtocolError
from videoqa_methods.density_selection import build_regions,source_time
from videoqa_budget32.selection import config,configure_visual_precision,SOURCE

FULL=ROOT/'outputs/full_videoqa/rd12_exp07__qwen32__videomme2700__20260930_r1'
BASE=ROOT/'outputs/full_videoqa/uniform_topk_rd12__qwen25vl__lmms_ref_v1__videomme2700__k32_20260924__r01'
MODEL=Path('/home/models/Salesforce/blip-itm-base-coco')
PYTHON='/App/conda/envs/explore_video_qa/bin/python'
CONDITIONS=('question_only','question_options')
# 已检查见证只供评价，不能传给评分函数；没有完整原因/关系标注的不冒充充分证据。
WITNESSES={
 '004-1':[('earliest_stage_text',[1],'直接阶段文字')],
 '006-3':[('nyc_parade',[89,91],'纽约游行文字及画面')],
 '001-1':[('tree_decoration',[43,48],'已见完整装饰树，不代表计数已验证')],
 '007-1':[('fire_scene',[12,13],'地面火堆及相邻动作状态')],
 '602-2':[('teutonic_1242',[197,198,199,200,201,202],'人物/战役/1242年部分见证，非完整覆亡证明')],
 '602-3':[('rail_length',[2265],'9289KM文字'),('franco_russian_context',[2232],'联盟背景，不等同贷款证据')],
 '607-2':[('hexagonal_lake',[641,642,733,734],'六角湖附近场景，不等同自然/人工河道关系')],
 '304-1':[('coat_pose_context',[64,118],'姿势上下文，不是已确认原因')],
 '307-1':[('year_73',[70],'73 BCE锚点，未完成事件映射'),('year_72',[175,200],'72 BCE锚点')],
}


def build_query(question,options):
    """接口：只接收原题和全部原选项，不接受答案、事实、题型或人工改写。"""
    if not isinstance(question,str) or not question.strip() or len(options)!=4 or any(not isinstance(x,str) or not x.strip() for x in options):
        raise ProtocolError('Expected question and four original options')
    return question+'\n'+'\n'.join(options)


def ranking(candidates,scores):
    """接口：采用原稳定同分规则；返回排名而不是回答正确概率。"""
    if len(candidates)!=len(scores) or not candidates or not np.isfinite(scores).all() or any(not 0<=x<=1 for x in scores):
        raise ProtocolError('Invalid score vector')
    return sorted(range(len(scores)),key=lambda i:(-scores[i],candidates[i]['source_pts'],candidates[i]['candidate_index']))


def prepare(run,gpus):
    """接口：问答前固定9题、查询、源码/模型/源资产及评分预算，评价材料另存。"""
    from transformers import BlipProcessor
    from videoqa_full.prepare import environment
    from videoqa_full.state import rows
    run=Path(run);run.mkdir(parents=True,exist_ok=True)
    with exclusive(run/'prepare.lock'):
        if (run/'protocol.json').exists():return verify(run)
        assert len(gpus)==len(set(gpus))==2 and set(gpus)<=set(map(str,range(8)))
        qs=read_json(FULL/'case_manifest.json')['unique_ids'];assert len(qs)==len(set(qs))==9 and set(qs)==set(WITNESSES)
        assert read_json(FULL/'phase_completion.json')['status']=='full_test_and_bounded_review_completed'
        old=read_json(SOURCE/'protocol.json');model_files={p:x['sha256'] for p,x in old['model_files'].items() if p.startswith(str(MODEL)+'/')}
        assert any(p.endswith('pytorch_model.bin') for p in model_files)
        for p,h in model_files.items():assert sha256(p)==h,p
        tokenizer=BlipProcessor.from_pretrained(MODEL,local_files_only=True).tokenizer
        limit=read_json(MODEL/'config.json')['text_config']['max_position_embeddings'];data={r['question_id']:r for r in rows()}
        inputs={};sources={str(FULL/'case_manifest.json'):sha256(FULL/'case_manifest.json')};n=0;batches=0
        # 步骤1：评分输入只保留问题/选项，不携带标准答案；候选固定为原粗前缀。
        for q in qs:
            row=data[q];pool_path=BASE/'pools'/(q+'.json');feature_path=BASE/'features'/(q+'.npy');check_path=BASE/'selection_checkpoints'/(q+'.json')
            check=read_json(check_path)
            for p in (pool_path,feature_path):assert sha256(p)==check['files'][str(p.relative_to(BASE))]
            for p in (pool_path,feature_path,check_path):sources[str(p)]=sha256(p)
            pool=read_json(pool_path);assert pool['initial_count']==len(pool['initial_scores'])
            video=pool['video'];video_id=row['video_id'];assert sha256(video['path'])==old['video_hashes'][video_id]
            sources[video['path']]=old['video_hashes'][video_id]
            texts=dict(question_only=row['question'],question_options=build_query(row['question'],row['options']))
            tokens={k:len(tokenizer(t,truncation=False)['input_ids']) for k,t in texts.items()};assert max(tokens.values())<=limit
            item=dict(question_id=q,question=row['question'],options=row['options'],texts=texts,tokens=tokens,
                pool_path=str(pool_path),feature_path=str(feature_path),initial_count=pool['initial_count'])
            durable(run/'inputs'/(q+'.json'),item);inputs[q]=sha256(run/'inputs'/(q+'.json'))
            n+=pool['initial_count'];batches+=(pool['initial_count']+15)//16
        # 步骤2：固定已见线索仅供事后评价；不新增补查、最终选帧或问答。
        durable(run/'evaluation.json',dict(witnesses=WITNESSES,scope='Listed partial witnesses only; no sufficiency or accuracy claim'))
        names=['src/videoqa_audit/query_input_probe.py','scripts/run_query_input_probe.py','src/videoqa_runtime/baseline_selection.py',
            'src/videoqa_methods/model_ops.py','src/videoqa_methods/density_selection.py','src/videoqa_budget32/selection.py',
            'src/videoqa_runtime/common.py','src/videoqa_full/state.py','configs/retrieval_density_v1.json']
        import shutil
        code={name:sha256(ROOT/name) for name in names}
        for name in code:
            dest=run/'code_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
        spec=dict(run_id=run.name,question_ids=qs,input_sha256=inputs,sources=sources,model_files=model_files,code_sha256=code,
            evaluation_sha256=sha256(run/'evaluation.json'),environment=environment(PYTHON),allowed_gpus=gpus,
            coarse_frames=n,normal_frame_text_scores=2*n,normal_batch_calls=2*batches,synthetic_batch_calls=4,
            conditions=list(CONDITIONS),batch_size=16,position_limit=limit,fp32_tolerance=1e-6,
            configuration=config(32,32),qa_calls=0,scope_classifier_calls=0,query_generation_calls=0,utc=utc())
        durable(run/'protocol.json',spec);return spec


def verify(run):
    """接口：恢复前校验身份，已开始而未确认返回的评分不自动重复。"""
    from videoqa_full.prepare import environment
    run=Path(run);spec=read_json(run/'protocol.json')
    assert spec['environment']==environment(PYTHON)
    for p,h in {**spec['sources'],**spec['model_files']}.items():assert sha256(p)==h,p
    for p,h in spec['code_sha256'].items():assert sha256(ROOT/p)==h,p
    for q,h in spec['input_sha256'].items():assert sha256(run/'inputs'/(q+'.json'))==h
    assert sha256(run/'evaluation.json')==spec['evaluation_sha256']
    return spec


def score_once(run,q,condition,start,scorer,encoded,images,gpu,rgb_hashes):
    """接口：一次FP32评分事务；完成批次按指纹恢复，不确定状态暂停，不隐式重试。"""
    path=run/'calls'/q/condition/(str(start)+'.json');features=path.with_suffix('.npy')
    identity=dict(question_id=q,condition=condition,start=start,frames=len(images),gpu=gpu,
        rgb_sha256=rgb_hashes,token_ids=encoded['input_ids'][0].tolist(),protocol_sha256=sha256(run/'protocol.json'))
    if path.exists():
        saved=read_json(path);assert saved['identity']==identity and saved['status']=='completed','Uncertain or changed BLIP call'
        assert sha256(features)==saved['feature_sha256']
        return saved['scores'],np.load(features,allow_pickle=False)
    # 步骤1：GPU调用前持久化；即使异常也保留已开始的成本与身份。
    durable(path,dict(status='started_may_forward',identity=identity,utc=utc()))
    scores,vectors,pre,forward=scorer.score_batch_features(encoded,images)
    # 步骤2：先保存返回产物，再确认事务完成；后处理崩溃不会静默再次前向。
    save_array(features,vectors)
    durable(path,dict(status='completed',identity=identity,scores=scores,feature_sha256=sha256(features),
        preprocessing_seconds=pre,forward_seconds=forward,utc=utc()))
    return scores,vectors


def score_question(run,q,scorer,gpu,spec):
    """接口：同一批RGB先后评分两文本；原题重放不符即停，不把模型差异误当查询效果。"""
    item=read_json(run/'inputs'/(q+'.json'));dest=run/'results'/(q+'.json')
    if dest.exists():
        result=read_json(dest);assert result['protocol_sha256']==sha256(run/'protocol.json')
        return result
    pool=read_json(item['pool_path']);coarse=pool['candidates'][:pool['initial_count']]
    frozen=np.load(item['feature_path'],allow_pickle=False,mmap_mode='r')[:len(coarse)]
    encoded={k:scorer.tokenize(item['texts'][k]) for k in CONDITIONS}
    scores={k:[] for k in CONDITIONS};seen=[];batch=[];rgb=[];metadata={};started=time.perf_counter();errors=[]
    def flush():
        begin=len(seen)-len(batch)
        if (run/'STOP.json').exists():raise ProtocolError('Peer stopped: inspect failure before resume')
        hashes=[hashlib.sha256(np.asarray(im.convert('RGB')).tobytes()).hexdigest() for im in batch];rgb.extend(hashes)
        original,v1=score_once(run,q,'question_only',begin,scorer,encoded['question_only'],batch,gpu,hashes)
        e_score=float(np.max(np.abs(np.asarray(original)-pool['initial_scores'][begin:begin+len(batch)])))
        e_feature=float(np.max(np.abs(v1-frozen[begin:begin+len(batch)])))
        assert max(e_score,e_feature)<=spec['fp32_tolerance'],('Historical replay mismatch',q,begin,e_score,e_feature)
        expanded,v2=score_once(run,q,'question_options',begin,scorer,encoded['question_options'],batch,gpu,hashes)
        e_invariance=float(np.max(np.abs(v1-v2)));assert e_invariance<=spec['fp32_tolerance']
        errors.append(dict(start=begin,score_error=e_score,feature_error=e_feature,question_invariance_error=e_invariance))
        scores['question_only'].extend(original);scores['question_options'].extend(expanded)
        for im in batch:im.close()
        batch.clear()
        durable(run/'workers'/(gpu+'.json'),dict(status='scoring',question_id=q,completed_frames=len(seen),pid=os.getpid(),utc=utc()))
    # 步骤1：线性有界解码与原粗网格逐帧校验，两个条件使用同一RGB，不常驻长视频。
    for row,frame in iter_candidate_frames(pool['video']['path'],metadata):
        expected=coarse[len(seen)]
        for key in ('candidate_index','source_pts','source_frame_index','requested_seconds'):
            assert row[key]==expected[key],(q,key,row,expected)
        seen.append(row);batch.append(frame.to_image())
        if len(batch)==16:flush()
    if batch:flush()
    assert len(seen)==len(coarse)
    # 步骤2：只计算Top32种子与原规则窗口，不进行新补查或最终32帧选择。
    plans={k:build_regions(coarse,scores[k],pool['video'],spec['configuration']) for k in CONDITIONS}
    assert plans['question_only']==pool['plan'],('Original discrete plan changed',q)
    result=dict(question_id=q,status='completed',protocol_sha256=sha256(run/'protocol.json'),gpu=gpu,scores=scores,plans=plans,
        ranking={k:ranking(coarse,scores[k]) for k in CONDITIONS},rgb_sha256=rgb,replay_errors=errors,
        new_qa_calls=0,frame_text_scores=2*len(coarse),seconds=time.perf_counter()-started,utc=utc())
    durable(dest,result);return result


def worker(run,gpu,questions):
    """接口：单卡常驻BLIP，只评分本批分配问题；异常落盘后停止同伴，不自动改条件。"""
    os.environ.update(CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    offline_environment();run=Path(run);spec=read_json(run/'protocol.json');active=None
    try:
        import torch
        from PIL import Image
        from videoqa_methods.model_ops import FeatureScorer
        torch.set_num_threads(2);torch.set_num_interop_threads(1);precision=configure_visual_precision(torch)
        started=time.perf_counter();scorer=FeatureScorer(str(MODEL));load=time.perf_counter()-started
        # 步骤1：合成预热独立记账，不产生开发/全量问题的回答。
        warmups=0
        for j in range(2):
            path=run/'synthetic'/f'{gpu}_{j}.json'
            if path.exists():
                assert read_json(path)['status']=='completed','Uncertain synthetic forward; inspect before resume'
                continue
            frames=[Image.new('RGB',(1280,720),'gray') for _ in range(16)]
            durable(path,dict(status='started',gpu=gpu,frames=16,utc=utc()))
            scorer.score_batch_features(scorer.tokenize('Which synthetic color is visible?'),frames)
            warmups+=1
            for im in frames:im.close()
            durable(path,dict(status='completed',gpu=gpu,frames=16,precision=precision,load_seconds=load,utc=utc()))
        durable(run/'resources'/f'{gpu}_{os.getpid()}.json',dict(gpu=gpu,pid=os.getpid(),load_seconds=load,
            new_synthetic_calls=warmups,recovery_without_new_warmup=warmups==0,utc=utc()))
        # 步骤2：每题独立原子结果，原始调用账本保持可复算。
        for q in questions:
            active=q
            if (run/'STOP.json').exists():raise ProtocolError('Peer failure')
            score_question(run,q,scorer,gpu,spec)
        durable(run/'workers'/(gpu+'.json'),dict(status='completed',pid=os.getpid(),question_ids=questions,utc=utc()))
    except BaseException as exc:
        error=dict(gpu=gpu,question_id=active,error=repr(exc),traceback=traceback.format_exc(),utc=utc())
        durable(run/'failures'/f'{gpu}_{time.time_ns()}.json',error);durable(run/'STOP.json',error);raise


def execute(run):
    """接口：脱离会话的两卡协调器，完成评分后自动汇总；不启动任何问答阶段。"""
    from videoqa_runtime.baseline_run import available_gpus
    run=Path(run)
    with exclusive(run/'coordinator.lock'):
        spec=verify(run)
        if len(list((run/'results').glob('*.json')))==9:return summarize(run)
        if (run/'STOP.json').exists():raise ProtocolError('Scoring paused; automatic retry prohibited')
        required=spec['allowed_gpus']
        while True:
            try:free,states=available_gpus(required)
            except RuntimeError:free,states=[],{}
            if set(required)<=set(free):break
            durable(run/'status.json',dict(status='waiting_resources',pid=os.getpid(),utc=utc()));time.sleep(15)
        # 步骤1：按粗候选量平衡整题任务，GPU只影响并行，不按对错分派。
        assignment={g:[] for g in required};load={g:0 for g in required}
        order=sorted(spec['question_ids'],key=lambda q:(-read_json(run/'inputs'/(q+'.json'))['initial_count'],q))
        for q in order:
            g=min(required,key=lambda g:(load[g],g));assignment[g].append(q);load[g]+=read_json(run/'inputs'/(q+'.json'))['initial_count']
        durable(run/'assignment.json',assignment);(run/'logs').mkdir(exist_ok=True);children={};logs=[];started=time.perf_counter()
        for gpu,qs in assignment.items():
            log=(run/'logs'/f'gpu_{gpu}.log').open('a');logs.append(log)
            children[gpu]=subprocess.Popen([PYTHON,str(ROOT/'scripts/run_query_input_probe.py'),'--run-id',run.name,'--stage','worker','--gpu',gpu,'--questions',*qs],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,cwd=ROOT)
        # 步骤2：只观察已知子进程；失败停止分派，不以观察超时重启。
        try:
            while any(p.poll() is None for p in children.values()):
                failed={g:p.returncode for g,p in children.items() if p.poll() not in (None,0)}
                if failed and not (run/'STOP.json').exists():durable(run/'STOP.json',dict(failed=failed,utc=utc()))
                durable(run/'status.json',dict(status='paused' if failed else 'running',completed=len(list((run/'results').glob('*.json'))),
                    total=9,pid=os.getpid(),sid=os.getsid(0),workers={g:p.pid for g,p in children.items()},seconds=time.perf_counter()-started,utc=utc()))
                time.sleep(3)
            assert all(p.returncode==0 for p in children.values()),'Worker failed; see STOP and failures'
        finally:
            for log in logs:log.close()
        result=summarize(run)
        durable(run/'status.json',dict(status='completed',completed=9,total=9,pid=os.getpid(),sid=os.getsid(0),seconds=time.perf_counter()-started,utc=utc()))
        return result


def summarize(run):
    """接口：完整9题后评估已列见证排名与窗口可达性，不把定位指标称为答案准确率。"""
    from fractions import Fraction
    run=Path(run);spec=verify(run);witnesses=read_json(run/'evaluation.json')['witnesses'];evaluation=[];questions=[]
    assert {p.stem for p in (run/'results').glob('*.json')}==set(spec['question_ids'])
    calls=list((run/'calls').glob('*/*/*.json'));assert len(calls)==spec['normal_batch_calls']
    assert all(read_json(p)['status']=='completed' for p in calls)
    assert sum(read_json(p)['identity']['frames'] for p in calls)==spec['normal_frame_text_scores']
    for q in spec['question_ids']:
        item=read_json(run/'inputs'/(q+'.json'));result=read_json(run/'results'/(q+'.json'));pool=read_json(item['pool_path']);coarse=pool['candidates'][:pool['initial_count']]
        assert result['protocol_sha256']==sha256(run/'protocol.json') and result['frame_text_scores']==2*len(coarse)
        conditions={}
        for condition in CONDITIONS:
            plan=result['plans'][condition];seeds=plan['seed_indices']
            union=sum(Fraction(r['right_fraction'])-Fraction(r['left_fraction']) for r in plan['regions'])
            conditions[condition]=dict(region_count=len(plan['regions']),window_union_seconds=float(union),
                refinement_eligible_regions=sum(r['refine'] for r in plan['regions']),
                unique_coarse_in_windows=len(set(i for r in plan['regions'] for i in r['coarse_indices'])),seeds=seeds)
            ranks={i:j+1 for j,i in enumerate(result['ranking'][condition])}
            for name,ids,meaning in witnesses[q]:
                best=min(ranks[i] for i in ids)
                reachable=any(Fraction(r['left_fraction'])<=source_time(coarse[i],pool['video'])<=Fraction(r['right_fraction']) for r in plan['regions'] for i in ids)
                evaluation.append(dict(question_id=q,witness=name,candidate_indices=ids,meaning=meaning,condition=condition,
                    best_rank=best,normalized_rank=(best-1)/max(1,len(coarse)-1),seed_hit=bool(set(ids)&set(seeds)),window_hit=reachable,
                    nearest_seed_seconds=min(abs(float(source_time(coarse[i],pool['video'])-source_time(coarse[j],pool['video']))) for i in ids for j in seeds)))
        questions.append(dict(question_id=q,candidates=len(coarse),tokens=item['tokens'],conditions=conditions,
            seed_overlap=len(set(result['plans']['question_only']['seed_indices'])&set(result['plans']['question_options']['seed_indices'])),
            max_replay_error=max(max(e['score_error'],e['feature_error'],e['question_invariance_error']) for e in result['replay_errors'])))
    summary=dict(status='completed',questions=questions,witnesses=evaluation,normal_frame_text_scores=spec['normal_frame_text_scores'],
        normal_batch_calls=len(calls),synthetic_batch_calls=len(list((run/'synthetic').glob('*.json'))),new_qa_calls=0,
        limitation='Only listed cues and proposed seed/windows; no new refinement, final selection or answer accuracy',utc=utc())
    durable(run/'summary.json',summary);return summary
