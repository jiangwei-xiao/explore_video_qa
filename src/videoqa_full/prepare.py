"""全量实验预检：独立冻结源资产、历史选帧与两套环境，不读答案驱动选择。"""
import os
import shutil
import subprocess
import time
from pathlib import Path
from videoqa_runtime.common import ROOT, read_json, sha256, offline_environment
from .state import (durable, utc, rows, digest, METHODS, MANIFEST, MANIFEST_HASH,
                    PYTHONS, QWEN_PATH, validate_frames, exclusive, pilot_ids)

R1='outputs/baselines/videomme_full_uniform_topk_20260916_r1'
R2='outputs/baselines/videomme_full_completion_20260917_r2'


def environment(python):
    """以实际解释器采集依赖版本，不将另一个环境的锁文件当作当前环境。"""
    import json
    code='import importlib.metadata as m,json; print(json.dumps(dict(sorted((d.metadata["Name"].lower(),d.version) for d in m.distributions()))))'
    return json.loads(subprocess.check_output([python,'-c',code],text=True))


def runtime_sources():
    """冻结所有本项目运行代码；文档另存，不因进度更新破坏运行指纹。"""
    paths=list((ROOT/'src').glob('**/*.py'))+list((ROOT/'scripts').glob('*.py'))
    paths += [ROOT/'configs/releases/rd2p_v1_0.json',ROOT/'configs/retrieval_density_v1.json',
              ROOT/'configs/local_model_inventory.json',ROOT/'configs/runtime_environment.json',
              ROOT/'configs/llava_source_manifest.json',ROOT/'configs/method_versions.json',ROOT/MANIFEST,
              ROOT/'data/manifests/video_mme_full_asset_audit.json',
              ROOT/'data/manifests/video_mme_development_200_20260920.json']
    return {str(p.relative_to(ROOT)):sha256(p) for p in sorted(set(paths))}


def verify_frozen_rd():
    """旧冻结代码/设计稿与运行协议不得因新增全量入口而改写。"""
    release=read_json(ROOT/'configs/releases/rd2p_v1_0.json')
    for name,h in release['code_sha256'].items():
        if sha256(ROOT/name)!=h: raise ValueError('RD frozen file changed: '+name)
    for path,hashkey in [('protocol_path','protocol_sha256'),('frozen_manifest_path','frozen_manifest_sha256')]:
        assert sha256(ROOT/release[path])==release[hashkey]
    return release


def verify_runtime(run,model):
    """每阶段启动/恢复时核对代码、环境、模型身份与所有导出选帧指纹。"""
    spec=read_json(Path(run)/'protocol.json')
    for name,h in spec['code_sha256'].items():
        if sha256(ROOT/name)!=h: raise ValueError('Run code changed: '+name)
    assert environment(PYTHONS[model])==spec['environments'][model]
    for path,entry in spec['model_files'].items():
        st=Path(path).stat()
        assert st.st_size==entry['bytes'] and st.st_mtime_ns==entry['mtime_ns']
    for name,h in spec['selection_exports'].items():
        assert sha256(Path(run)/name)==h
    return spec


def prepare(run):
    """接口：只执行CPU资产/协议检查；不调用任何GPU或真实问答。"""
    from videoqa_runtime.baseline_records import validate_result
    run=Path(run)
    run.mkdir(parents=True,exist_ok=True)
    with exclusive(run/'prepare.lock'):
        if (run/'protocol.json').exists():
            verify_runtime(run,'llava');verify_runtime(run,'qwen25vl');return
        started=time.perf_counter();data=rows();release=verify_frozen_rd()
        durable(run/'preparation_status.json',dict(status='checking_assets',utc=utc()))
        # 步骤1：900视频逐一哈希，不因每视频三题重复扫描资产。
        audit=read_json(ROOT/'data/manifests/video_mme_full_asset_audit.json')
        expected={f['path']:f['sha256'] for f in audit['files']}
        videos={};video_stats={}
        for row in data:
            if row['video_id'] in videos:continue
            name='data/'+row['video_relative_path'];p=ROOT/name;h=sha256(p)
            assert h==expected[name]
            videos[row['video_id']]=h
            video_stats[row['video_id']]=dict(path=name,bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
        print('900 video hashes verified',flush=True)
        # 步骤2：共享模型只读，旧模型按库存哈希复验，Qwen首次建立完整文件清单。
        inventory=read_json(ROOT/'configs/local_model_inventory.json');model_files={}
        for model in inventory['models'].values():
            for name,entry in model['files'].items():
                p=Path(model['path'])/name;h=sha256(p);assert h==entry['sha256']
                model_files[str(p)]=dict(sha256=h,bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
        for p in sorted(Path(QWEN_PATH).iterdir()):
            if p.is_file():model_files[str(p)]=dict(sha256=sha256(p),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
        print('LLaVA / BLIP / Qwen model hashes verified',flush=True)
        # 步骤3：两轮历史基线互斥合并，逐条验证后只导出源帧，不携带历史答案。
        specs={r:read_json(ROOT/r/'protocol.json') for r in (R1,R2)}
        ph={r:sha256(ROOT/r/'protocol.json') for r in (R1,R2)}
        source_hashes={};exports={};correct={m:0 for m in METHODS[:2]}
        historical=[]
        for source in (R1,R2):
            for name,h in specs[source]['code_sha256'].items():
                assert sha256(ROOT/source/'code_snapshot'/name)==h
            source_hashes[source+'/protocol.json']=ph[source]
        for row in data:
            qid=row['question_id']
            for method,key in zip(METHODS[:2],('uniform','topk')):
                candidates=[ROOT/r/'results'/key/(qid+'.json') for r in (R1,R2)]
                paths=[p for p in candidates if p.exists()];assert len(paths)==1
                path=paths[0];source=R1 if str(path).startswith(str(ROOT/R1)+'/') else R2
                record=read_json(path);validate_result(record,row,ph[source],specs[source])
                assert record['source_video_sha256']==videos[row['video_id']]
                h=sha256(path);source_hashes[str(path.relative_to(ROOT))]=h
                selected=dict(video=record['selection']['video'],selected_frames=record['selection']['selected_frames'])
                validate_frames(selected)
                payload=dict(question_id=qid,video_id=row['video_id'],method=method,
                    question_sha256=digest([row['question'],row['options']]),source_video_sha256=videos[row['video_id']],
                    source_path=str(path.relative_to(ROOT)),source_sha256=h,selection=selected)
                destination=run/'selections'/method/(qid+'.json');durable(destination,payload)
                exports[str(destination.relative_to(run))]=sha256(destination)
                correct[method]+=record['correct']
                historical.append(dict(question_id=qid,video_id=row['video_id'],stratum=row['stratum'],model='llava',
                    method=method,correct=record['correct'],parsed_answer=record['answer']['parsed_answer'],
                    raw_output=record['answer']['raw_output'],source_path=str(path.relative_to(ROOT)),source_sha256=h,
                    timings=record['timings'],memory=record['memory']))
        assert correct=={'BASE-Uniform':1572,'BASE-BLIP-TopK':1557}
        durable(run/'historical_baselines.json',historical)
        exports['historical_baselines.json']=sha256(run/'historical_baselines.json')
        # 步骤4：检查全部问题BLIP长度，禁止截断；记录两环境依赖及源代码快照。
        offline_environment()
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(inventory['models']['blip']['path'],local_files_only=True)
        lengths=[len(tokenizer(r['question'],truncation=False)['input_ids']) for r in data]
        assert max(lengths)<=512
        envs={m:environment(p) for m,p in PYTHONS.items()}
        assert envs['llava']['transformers']=='4.45.2' and envs['qwen25vl']['transformers']=='4.50.0'
        for env in envs.values():
            assert env['torch'].startswith('2.6.0') and env['torchvision'].startswith('0.21.0')
        code=runtime_sources()
        for name in code:
            dest=run/'code_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
        dev=read_json(ROOT/'data/manifests/video_mme_development_200_20260920.json')['rows']
        assert len({r['video_id'] for r in dev})==200
        old=ROOT/'outputs/methods/density_e1_200_20260922_r1'
        historical_rd={r['question_id']:sha256(old/'results/e1'/(r['question_id']+'.json')) for r in dev}
        spec=dict(version='videomme-full-rd12-qwen-video-v1',run_id=run.name,created_at=utc(),
            manifest=MANIFEST,manifest_sha256=MANIFEST_HASH,question_count=2700,video_count=900,
            models=list(PYTHONS),methods={'llava':['RD-1.2'],'qwen25vl':list(METHODS)},
            max_answer_calls_per_group=2700,total_answer_calls=10800,automatic_generation_retries=0,
            pilot_question_ids=pilot_ids(data),config=release['config'],
            llava_inference=release['inference'],qwen_inference=dict(model_path=QWEN_PATH,input='video',
                dtype='bfloat16',attention='sdpa',min_pixels=128*28*28,max_pixels=768*28*28,
                encoding_fps=2.0,true_pts_in_text=True,internal_padding='repeat last frame only if odd',
                max_new_tokens=8,seed=2027,template='official checkpoint chat template'),
            code_sha256=code,environments=envs,model_files=model_files,video_hashes=videos,video_stats=video_stats,
            source_results=source_hashes,selection_exports=exports,historical_rd=historical_rd,
            blip_text_lengths=lengths,prepared_seconds=time.perf_counter()-started,
            timing='llava raw E2E; qwen cached selection + final decode + QA, not raw E2E',
            temporal_limit='Qwen uniform encoding grid is not irregular true PTS; true times remain in text',
            available_disk_bytes=shutil.disk_usage(run).free)
        assert spec['available_disk_bytes']>200*1024**3
        durable(run/'protocol.json',spec)
        durable(run/'preparation_status.json',dict(status='prepared',seconds=time.perf_counter()-started,utc=utc()))
        print('Prepared 2700 questions, 5400 source selections, 10800 answer IDs',flush=True)
