"""exp23后台纯视觉特征与选择准备；批次检查点恢复，不调用最终问答。"""
import os
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
import argparse
import importlib.metadata
import multiprocessing as mp
from pathlib import Path
import shutil
import sys
import time
import traceback
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json,sha256,offline_environment
from videoqa_full.state import durable,exclusive,save_array,utc
from videoqa_audit.query_input_probe import BASE
from videoqa_budget32.frames import FrameProvider
from videoqa_methods.optional_region_selection import select_optional_regions

SOURCE=ROOT/'outputs/analysis/optional_seed64_pool200_20261004_r1'
MODEL=Path('/home/models/Salesforce/blip-itm-base-coco')
METHOD='RD-1.2-exp23'


def visual_once(scorer,selection,run,key,gpu,fingerprint):
    """接口：精确解码后单次视觉前向；已完成批次复用，未知批次禁止重试。"""
    call=run/'calls'/(key+'.json');array=run/'batch_features'/(key+'.npy')
    if call.exists():
        record=read_json(call)
        if record['status']!='completed' or record['fingerprint']!=fingerprint or record['gpu']!=gpu:
            raise ValueError('未知视觉前向或GPU身份不一致，不重试')
        if record['frames']!=selection['selected_frames']:raise ValueError('缓存批次源帧不一致')
        if sha256(array)!=record['feature_sha256']:raise ValueError('批次特征损坏')
        return np.load(array,allow_pickle=False),record
    # 步骤1：完整源PTS映射已经冻结，读取时仍由精确帧提供器约束，最多16张。
    start=time.perf_counter();images,hashes=FrameProvider(selection).decode();decode=time.perf_counter()-start
    if not 1<=len(images)<=16:raise ValueError('视觉批量超限')
    identity=dict(status='started',fingerprint=fingerprint,gpu=gpu,frames=selection['selected_frames'],
                  rgb_sha256=hashes,count=len(images),utc=utc())
    durable(call,identity)
    # 步骤2：特征返回后先原子保存，再确认批次完成；不为失败结果追加调用。
    try:vectors,pre,forward=scorer.visual_batch(images)
    finally:
        for image in images:image.close()
    if vectors.dtype!=np.float32 or not np.isfinite(vectors).all():raise ValueError('视觉特征精度异常')
    digest=save_array(array,vectors)
    record=dict(identity,status='completed',feature_sha256=digest,decode_seconds=decode,
                preprocessing_seconds=pre,forward_seconds=forward)
    durable(call,record);return vectors,record


def worker(run_path,gpu,questions,stop):
    """接口：单卡常驻BLIP，只做合成验收、粗特征控制和新细帧视觉编码。"""
    os.environ['CUDA_VISIBLE_DEVICES']=gpu
    run=Path(run_path);active=None
    try:
        offline_environment()
        import torch
        from videoqa_methods.density_pipeline import DensityScorer
        from videoqa_budget32.selection import configure_visual_precision
        torch.set_num_threads(2);torch.set_num_interop_threads(1);precision=configure_visual_precision(torch)
        fingerprint=sha256(run/'protocol.json')
        load_start=time.perf_counter();scorer=DensityScorer(str(MODEL))
        durable(run/'worker_loads'/f'{gpu}_{os.getpid()}.json',dict(gpu=gpu,pid=os.getpid(),
            model_load_seconds=time.perf_counter()-load_start,model_memory_bytes=torch.cuda.memory_allocated(),precision=precision))
        syn=run/'synthetic'/(gpu+'.json')
        # 步骤1：完成验收不重复前向；首次调用开始即记账，异常留待核查。
        if syn.exists():
            old=read_json(syn)
            if old['status']!='completed' or old['fingerprint']!=fingerprint:raise ValueError('合成状态不确定')
        else:
            durable(syn,dict(status='started',fingerprint=fingerprint,gpu=gpu))
            result=scorer.synthetic_acceptance()
            durable(syn,dict(status='completed',fingerprint=fingerprint,gpu=gpu,result=result,precision=precision))
        first=questions[0];old=read_json(BASE/'pools'/(first+'.json'))
        old_vectors=np.load(BASE/'features'/(first+'.npy'),allow_pickle=False)
        count=min(16,old['initial_count'])
        vectors,_=visual_once(scorer,dict(video=old['video'],selected_frames=old['candidates'][:count]),
                              run,'control/'+gpu,gpu,fingerprint)
        error=float(np.max(np.abs(vectors-old_vectors[:count])))
        if error>1e-6:raise ValueError('粗帧纯视觉控制误差超限')
        durable(run/'control_acceptance'/(gpu+'.json'),dict(status='passed',gpu=gpu,question_id=first,
            fingerprint=fingerprint,maximum_abs_error=error,tolerance=1e-6,count=count))
        durable(run/'workers'/(gpu+'.json'),dict(status='ready',pid=os.getpid(),gpu=gpu,control_error=error,precision=precision))
        # 步骤2：静态GPU归属支持部分恢复；旧粗向量复用，全部细向量在本轮重新计算。
        for q in questions:
            if stop.is_set():return
            active=q;done=run/'checkpoints'/(q+'.json')
            if done.exists():continue
            pool=read_json(SOURCE/'pools'/(q+'.json'));n=pool['initial_count']
            coarse=np.load(BASE/'features'/(q+'.npy'),allow_pickle=False)[:n].copy()
            blocks=[coarse];batch_records=[]
            for offset in range(n,len(pool['candidates']),16):
                if stop.is_set():return
                frames=pool['candidates'][offset:offset+16]
                vectors,record=visual_once(scorer,dict(video=pool['video'],selected_frames=frames),
                    run,'fine/'+q+'/'+str(offset-n),gpu,fingerprint)
                if vectors.shape[1]!=coarse.shape[1]:raise ValueError('特征维度不一致')
                blocks.append(vectors);batch_records.append(record)
            features=np.concatenate(blocks,axis=0)
            start=time.perf_counter();selected,trace=select_optional_regions(pool,features,32)
            selection_seconds=time.perf_counter()-start
            selection=dict(video=pool['video'],selected_frames=[pool['candidates'][i] for i in selected])
            images,hashes=FrameProvider(selection).decode()
            for im in images:im.close()
            fp=run/'features'/(q+'.npy');feature_hash=save_array(fp,features)
            sp=run/'selections'/(q+'.json');tp=run/'traces'/(q+'.json')
            durable(tp,trace)
            durable(sp,dict(question_id=q,method=METHOD,selection=selection,selected_indices=selected,rgb_sha256=hashes,
                            source_pool_sha256=sha256(SOURCE/'pools'/(q+'.json')),feature_sha256=feature_hash))
            durable(done,dict(question_id=q,gpu=gpu,fingerprint=fingerprint,files={str(fp):feature_hash,str(sp):sha256(sp),str(tp):sha256(tp)},
                fine_frames=sum(r['count'] for r in batch_records),fine_batches=len(batch_records),
                selection_seconds=selection_seconds,zero_regions=len(trace['zero_quota_regions'])))
            durable(run/'workers'/(gpu+'.json'),dict(status='running',question_id=q,gpu=gpu,pid=os.getpid(),utc=utc()))
        durable(run/'workers'/(gpu+'.json'),dict(status='completed',gpu=gpu,pid=os.getpid(),utc=utc(),
            actual_forward_counts=scorer.counts,peak_memory_bytes=torch.cuda.max_memory_allocated()))
    except BaseException as exc:
        stop.set();traceback.print_exc()
        durable(run/'failures'/f'{gpu}_{time.time_ns()}.json',dict(error=repr(exc),question_id=active,gpu=gpu,pid=os.getpid(),utc=utc()))
        raise


def main(run):
    """接口：冻结源码/资产/调用上限，后台准备全部200题，不自动进入问答。"""
    from videoqa_runtime.baseline_run import available_gpus
    with exclusive(run/'run.lock'):
        started=time.perf_counter();resumed=(run/'protocol.json').exists()
        durable(run/'status.json',dict(status='verifying',pid=os.getpid(),sid=os.getsid(0),utc=utc()))
        source=read_json(SOURCE/'summary.json')
        if len(source['records'])!=200 or len({r['question_id'] for r in source['records']})!=200:
            raise ValueError('准备名单不唯一或分母错误')
        if source['config']['seed_budget']!=64 or source['config']['frame_budget']!=32:
            raise ValueError('不是冻结M64/K32配置')
        if source['total_fine']!=19918 or source['maximum_visual_batches']!=1338:
            raise ValueError('上游视觉预算变化')
        for path,digest in source['sources'].items():
            if sha256(path)!=digest:raise ValueError('上游来源变化: '+path)
        for row in source['records']:
            if sha256(SOURCE/'pools'/(row['question_id']+'.json'))!=row['pool_sha256']:raise ValueError('池指纹变化')
        sources={str(SOURCE/'summary.json'):sha256(SOURCE/'summary.json')}
        for path in [Path(__file__),ROOT/'src/videoqa_methods/optional_region_selection.py',ROOT/'src/videoqa_methods/density_pipeline.py',
                     ROOT/'src/videoqa_budget32/frames.py',ROOT/'src/videoqa_methods/region_context.py',ROOT/'src/videoqa_methods/weighted_seed_support.py',
                     ROOT/'src/videoqa_methods/coarse_eligibility.py',ROOT/'src/videoqa_methods/density_selection.py',
                     ROOT/'src/videoqa_methods/model_ops.py',ROOT/'src/videoqa_runtime/baseline_selection.py',ROOT/'src/videoqa_budget32/selection.py']:
            sources[str(path)]=sha256(path)
        for path in MODEL.iterdir():
            if path.is_file():sources[str(path)]=sha256(path)
        pp=run/'protocol.json'
        # 步骤1：GPU分配在首次运行固定，恢复不得把未完成批次迁到其他物理卡。
        if pp.exists():
            spec=read_json(pp)
            if spec['sources']!=sources:raise ValueError('本轮代码或模型变化')
        else:
            free,_=available_gpus(list(map(str,range(8))))
            if not free:raise RuntimeError('没有空闲GPU，尚未启动模型')
            ids=[r['question_id'] for r in source['records']]
            ids.sort(key=lambda q:-read_json(SOURCE/'pools'/(q+'.json'))['video']['duration_seconds'])
            routes={gpu:ids[i::len(free)] for i,gpu in enumerate(free)}
            spec=dict(method=METHOD,sources=sources,routes=routes,question_count=200,
                max_fine_visual_frames=19918,max_fine_visual_batches=1338,max_control_frames=16*len(free),
                synthetic_batches_per_gpu=3,new_real_itm_calls=0,max_qa_calls=0,
                dependencies={n:importlib.metadata.version(n) for n in ['torch','transformers','numpy','Pillow','av']})
            durable(pp,spec)
            for path,digest in sources.items():
                p=Path(path)
                if p.is_relative_to(ROOT) and p.relative_to(ROOT).parts[0] in ['src','scripts']:
                    dest=run/'code_snapshot'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
        fingerprint=sha256(pp)
        actual_dependencies={n:importlib.metadata.version(n) for n in spec['dependencies']}
        if actual_dependencies!=spec['dependencies']:raise ValueError('依赖版本变化')
        for directory in ['calls','synthetic']:
            for path in (run/directory).rglob('*.json'):
                record=read_json(path)
                if record['status']!='completed' or record['fingerprint']!=fingerprint:
                    raise ValueError('存在未确认前向，禁止自动恢复模型调用')
        for path in (run/'checkpoints').glob('*.json'):
            saved=read_json(path)
            if saved['fingerprint']!=fingerprint:raise ValueError('检查点身份变化')
            for p,h in saved['files'].items():
                if sha256(p)!=h:raise ValueError('检查点文件变化')
        required=[gpu for gpu,qs in spec['routes'].items() if any(not (run/'checkpoints'/(q+'.json')).exists() for q in qs)]
        # 步骤2：已完成时不加载模型，未完成时只用原归属的空闲卡，不抢占。
        if required:
            free,_=available_gpus(required)
            if set(free)!=set(required):raise RuntimeError('原归属GPU被占用，等待后恢复')
            context=mp.get_context('spawn');stop=context.Event();processes=[]
            for gpu in required:
                p=context.Process(target=worker,args=(str(run),gpu,spec['routes'][gpu],stop));p.start();processes.append(p)
            while any(p.is_alive() for p in processes):
                durable(run/'status.json',dict(status='running',completed=len(list((run/'checkpoints').glob('*.json'))),
                    total=200,pid=os.getpid(),sid=os.getsid(0),utc=utc()))
                time.sleep(2)
            for p in processes:p.join()
            if stop.is_set() or any(p.exitcode!=0 for p in processes):raise RuntimeError('准备工作进程失败，未自动重试')
        # 步骤3：全量计数与恢复边界核验；此完成仅表示选择准备，不是问答完成。
        records=[read_json(run/'checkpoints'/(q+'.json')) for qs in spec['routes'].values() for q in qs]
        if len(records)!=200 or sum(r['fine_frames'] for r in records)!=19918 or sum(r['fine_batches'] for r in records)!=1338:
            raise ValueError('准备规模未闭合')
        for path,digest in source['sources'].items():
            if sha256(path)!=digest:raise ValueError('运行期来源变化')
        for path,digest in spec['sources'].items():
            if sha256(path)!=digest:raise ValueError('运行期源码或模型变化')
        fine_calls=[read_json(p) for p in (run/'calls/fine').rglob('*.json')]
        if len(fine_calls)!=1338 or sum(r['count'] for r in fine_calls)!=19918:
            raise ValueError('真实视觉调用账本不闭合')
        controls=[read_json(p) for p in (run/'control_acceptance').glob('*.json')]
        if len(controls)!=len(spec['routes']):raise ValueError('粗特征控制验收缺失')
        summary=dict(status='passed',questions=200,fine_frames=19918,fine_batches=1338,qa_calls=0,
            control_frames=sum(read_json(p)['count'] for p in (run/'calls/control').glob('*.json')),
            maximum_control_error=max(r['maximum_abs_error'] for r in controls),
            synthetic_workers=len(list((run/'synthetic').glob('*.json'))),
            zero_region_total=sum(r['zero_regions'] for r in records),
            files={str(p):sha256(p) for p in (run/'checkpoints').glob('*.json')})
        output=run/'summary.json'
        if output.exists() and read_json(output)!=summary:raise ValueError('完成汇总不一致')
        if not output.exists():durable(output,summary)
        durable(run/'attempts'/f'{time.time_ns()}.json',dict(resumed=resumed,wall_seconds=time.perf_counter()-started,
            worker_gpus=required,completed=200,qa_calls=0))
        durable(run/'status.json',dict(status='completed',completed=200,total=200,pid=os.getpid(),sid=os.getsid(0),utc=utc()))
        print(summary,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True,type=Path);args=parser.parse_args()
    args.run.mkdir(parents=True,exist_ok=True)
    try:main(args.run)
    except BaseException as exc:
        durable(args.run/'status.json',dict(status='paused',error=repr(exc),pid=os.getpid(),sid=os.getsid(0),utc=utc()))
        raise
