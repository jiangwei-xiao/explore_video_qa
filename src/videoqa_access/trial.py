"""共享预览、六策略及有限原片访问的固定试点；不创建新RD算法版本。"""
from collections import Counter
from fractions import Fraction
import hashlib
import importlib.util
import io
import os
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image
from videoqa_runtime.common import ROOT, read_json, sha256
from videoqa_full.state import durable, digest, save_array, utc
from .transport import AccessClient, preview_bytes, rgb_hash, pack_frames

METHODS=('BASE-Uniform','BASE-BLIP-TopK','REF-WFS-SB','CTRL-WFS-preview','CTRL-TopK-local','RD-1.2-exp12')
DEV=ROOT/'data/manifests/video_mme_development_200_20260920.json'
INDEX=ROOT/'outputs/analysis/lmms_bridge_videomme_20260923_r1'
ML=ROOT/'outputs/full_videoqa/mlvu_dev__S1__llava8_qwen32__four_methods__20261014_r1'
MLDEV=ROOT/'outputs/analysis/dual_dataset_selection_review_20261015_r2/mlvu_development_210.json'
BLIP='/home/models/Salesforce/blip-itm-base-coco'


def freeze(path, value):
    """接口：冻结文件只创建一次，恢复必须逐字段相同。"""
    path=Path(path)
    if path.exists():
        if read_json(path)!=value:raise ValueError('Frozen record differs: '+str(path))
    else:durable(path,value)


def subset(data, field, groups, n):
    """接口：仅按原清单顺序取各类前n个不同视频；不读取答案或历史表现。"""
    counts=Counter();seen=set();result=[]
    # 步骤1扫描原冻结顺序；步骤2保存原顺序，禁止按题号/对错重新排列。
    for row in data:
        g=row[field]
        if g in groups and counts[g]<n and row['video_id'] not in seen:
            result.append(row);seen.add(row['video_id']);counts[g]+=1
    if any(counts[g]!=n for g in groups):raise ValueError('Insufficient fixed subset; no reserve substitution')
    return result


def coarse_frames(pts, video):
    """接口：以真实完整PTS映射1 FPS、1/4秒相位，去重且保留原目标。"""
    rows=[];seen=set();target=Fraction(1,4);tb=Fraction(video['time_base']);origin=video['start_pts']
    while target<Fraction(video['duration_fraction']):
        i=int(np.searchsorted(pts, Fraction(origin)+target/tb, side='left'))
        if i<len(pts) and i not in seen:
            p=int(pts[i]);t=(p-origin)*tb
            rows.append(dict(candidate_index=len(rows),source_frame_index=i,source_pts=p,
                source_seconds=float(p*tb),timestamp_seconds=float(t),requested_seconds=float(target)))
            seen.add(i)
        target+=1
    return rows


def targets_to_frames(targets, index, known):
    """接口：冻结时钟映射细目标；真实PTS必须在原范围，重复源不重复请求。"""
    pts=np.load(index['pts_path'],allow_pickle=False);v=index['video'];tb=Fraction(v['time_base']);seen=set(known);out=[]
    # 步骤1按时间映射首个不早于目标的源帧；步骤2核对真实源时间与目标窗口。
    for target,left,right,rid in sorted(targets,key=lambda x:(x[0],x[3])):
        i=int(np.searchsorted(pts,Fraction(v['start_pts'])+target/tb,side='left'))
        if i>=len(pts):continue
        p=int(pts[i]);t=(p-v['start_pts'])*tb
        if p in seen or not left<=t<=right:continue
        seen.add(p);out.append(dict(candidate_index=0,source_frame_index=i,source_pts=p,
            source_seconds=float(p*tb),timestamp_seconds=float(t),requested_seconds=float(target),
            requested_fraction=str(target),region_id=rid))
    return out


def prepare_manifest(root):
    """接口：先冻结60/70题与计分侧标签；输入侧清单不包含正确答案。"""
    from videoqa_mlvu.data import rows as mlrows, TASKS
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    if (root/'manifest.json').exists():
        m=read_json(root/'manifest.json')
        for p,h in m['sources'].items():
            if sha256(p)!=h:raise ValueError('Source manifest changed')
        return m
    a=subset(read_json(DEV)['rows'],'stratum',('short','medium','long'),20)
    bank={r['question_id']:r for r in mlrows(ML)};frozen_ml=read_json(MLDEV)['rows']
    b=subset([bank[r['question_id']] for r in frozen_ml],'task_type',TASKS,10)
    public=[];labels={};indices={};sources={str(p):sha256(p) for p in (DEV,MLDEV,ML/'manifest.json')}
    # 步骤1选择过程已结束后才投影计分字段；旧开发/预留清单均不改变。
    for dataset,data in [('videomme',a),('mlvu',b)]:
        for row in data:
            row=dict(row);q=row['question_id'];labels[dataset+'/'+q]=row.pop('answer_index')
            if 'original' in row:row['original']={k:v for k,v in row['original'].items() if k!='answer'}
            row.update(dataset=dataset);public.append(row);vid=row['video_id']
            if dataset=='videomme':
                ip=INDEX/'uniform_records'/(vid+'.json');old=read_json(ip);pp=INDEX/old['pts_file']
                record=dict(video=old['video'],pts_path=str(pp),pts_sha256=old['pts_sha256'],video_sha256=old['source_video_sha256'])
            else:
                ip=ML/'indices'/(vid+'.json');record=read_json(ip);pp=Path(record['pts_path'])
            if sha256(pp)!=record['pts_sha256']:raise ValueError('Frozen complete PTS changed')
            sources[str(ip)]=sha256(ip);sources[str(pp)]=sha256(pp)
            st=Path(record['video']['path']).stat()
            indices[vid]=dict(record,video_stat=dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns))
    freeze(root/'labels.json',labels)
    value=dict(rows=public,indices=indices,sources=sources,counts=dict(videomme=60,mlvu=70),selection='original order, duration20/task10, unique video; no outcome filtering')
    freeze(root/'manifest.json',value);return value


def build_preview(job):
    """接口：全片一次真实解码建立关键帧元数据与JPEG预览；模型调用0。"""
    import av
    from .transport import CountingReader
    root,vid,index=Path(job[0]),job[1],job[2];dest=root/'indices'/(vid+'.json')
    if dest.exists():
        old=read_json(dest)
        for p,h in old['preview_files'].items():
            if sha256(root/p)!=h:raise ValueError('Preview changed')
        return old
    path=Path(index['video']['path']);st=path.stat()
    if dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns)!=index['video_stat'] or sha256(path)!=index['video_sha256']:
        raise ValueError('Frozen original video changed')
    pts=np.load(index['pts_path'],allow_pickle=False);coarse=coarse_frames(pts,index['video'])
    wanted={f['source_pts']:f for f in coarse};records={};files={};keys=[];encoded=[];metadata=[]
    reader=CountingReader(path);began=time.perf_counter();count=0
    # 步骤1完整解码核对既有PTS；步骤2只保存预览，不缓存原图或历史特征。
    try:
        with av.open(reader) as container:
            stream=container.streams.video[0];stream.codec_context.thread_count=2
            for i,frame in enumerate(container.decode(stream)):
                if i>=len(pts) or frame.pts!=int(pts[i]):raise ValueError('Full source PTS/index differs')
                count+=1
                if frame.key_frame:keys.append(int(frame.pts))
                if frame.pts not in wanted:continue
                image=frame.to_image();data=preview_bytes(image);image.close()
                p=root/'previews'/vid/(str(frame.pts)+'.jpg');p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
                with Image.open(io.BytesIO(data)) as im:h=rgb_hash(im)
                files[str(p.relative_to(root))]=sha256(p)
                records[frame.pts]=dict(wanted[frame.pts],preview_file=str(p.relative_to(root)),preview_rgb_sha256=h)
        if count!=len(pts) or len(records)!=len(coarse) or not keys:raise ValueError('Incomplete source/index')
        for f in coarse:
            rec=records[f['source_pts']];encoded.append((root/rec['preview_file']).read_bytes())
            metadata.append(dict(f,rgb_sha256=rec['preview_rgb_sha256']))
        payload=pack_frames(dict(video_id=vid,frames=metadata,timing={}),encoded)
        key=hashlib.sha256(payload).hexdigest();pd=root/'transport_payloads';pd.mkdir(exist_ok=True)
        (pd/(key+'.bin')).write_bytes(payload)
        result=dict(index,keyframe_pts=keys,coarse=[records[f['source_pts']] for f in coarse],preview_files=files,
            preview_payload_sha256=key,preview_payload_bytes=len(payload),
            build_seconds=time.perf_counter()-began,decoded_frames=count,file_read_bytes=reader.bytes_read)
        freeze(dest,result);return result
    finally:reader.close()


def read_previews(root,index):
    """接口：只读取共享JPEG预览；返回的实际像素须符合预览发布指纹。"""
    images=[]
    for f in index['coarse']:
        with Image.open(Path(root)/f['preview_file']) as im:value=im.convert('RGB')
        if rgb_hash(value)!=f['preview_rgb_sha256']:raise ValueError('Preview RGB changed')
        images.append(value)
    return images


def scorer_factory(root):
    """接口：沿用FP32官方BLIP，逐批先记账后前向，失败或不确定前向禁止重发。"""
    from videoqa_methods.density_pipeline import DensityScorer
    class Scorer(DensityScorer):
        def _cached(self,encoded,images,visual):
            text={k:v.tolist() for k,v in (encoded or {}).items()};key=digest(dict(rgb=[rgb_hash(x) for x in images],text=text,visual=visual))
            cp=Path(root)/'blip_calls'/(key+'.json');array=cp.with_suffix('.npz')
            if cp.exists():
                rec=read_json(cp)
                if rec['status']!='completed' or sha256(array)!=rec['array_sha256']:raise RuntimeError('Uncertain BLIP forward')
                data=np.load(array,allow_pickle=False)
                return (data['features'].copy(),rec['preprocess_seconds'],rec['forward_seconds']) if visual else (data['scores'].tolist(),data['features'].copy(),rec['preprocess_seconds'],rec['forward_seconds'])
            durable(cp,dict(status='started',kind='visual' if visual else 'itm',frames=len(images),utc=utc()))
            out=super().visual_batch(images) if visual else super().score_batch_features(encoded,images)
            values=dict(features=out[0]) if visual else dict(scores=np.asarray(out[0]),features=out[1])
            array.parent.mkdir(parents=True,exist_ok=True);tmp=array.with_suffix('.part')
            with tmp.open('wb') as stream:np.savez(stream,**values)
            tmp.replace(array);pre,fw=out[-2:]
            durable(cp,dict(status='completed',kind='visual' if visual else 'itm',frames=len(images),array_sha256=sha256(array),preprocess_seconds=pre,forward_seconds=fw,utc=utc()))
            return out
        def score_batch_features(self,encoded,images):return self._cached(encoded,images,False)
        def visual_batch(self,images):return self._cached(None,images,True)
    import torch
    torch.manual_seed(2027);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    return Scorer(BLIP)


def encode(scorer, images, question=None):
    """接口：固定batch16；细帧先经过同一JPEG预览规格再视觉编码，ITM按方法分别计费。"""
    vectors=[];scores=[];pre=fw=0.;encoded=scorer.tokenize(question) if question is not None else None
    for offset in range(0,len(images),16):
        batch=images[offset:offset+16]
        if question is None:v,p,f=scorer.visual_batch(batch)
        else:s,v,p,f=scorer.score_batch_features(encoded,batch);scores.extend(s)
        vectors.append(v);pre+=p;fw+=f
    return scores,np.concatenate(vectors) if vectors else np.empty((0,768),np.float32),dict(preprocess_seconds=pre,forward_seconds=fw,frames=len(images),itm_frames=len(images) if question is not None else 0)


def wfs_indices(scores,features):
    """接口：执行固定公开WFS核心，K8及其默认配置，不使用发布选帧或答案。"""
    code=ROOT/'third_party/WFS-SB-reference/core.py'
    if sha256(code)!=read_json(ROOT/'configs/method_v1.json')['wfs_reference_sha256']:raise ValueError('WFS core changed')
    import sys
    spec=importlib.util.spec_from_file_location('access_wfs_fixed',code);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    obj=module.WFS(module.WFSConfig());n=len(scores)
    ids=list(map(int,obj.select_keyframes(np.asarray(scores),min(8,n),module.compute_dwt_level(n),module.compute_min_peak_distance(n),features)))
    if len(ids)!=min(8,n) or len(set(ids))!=len(ids):raise ValueError('WFS budget/identity changed')
    return sorted(ids)


def prepare_question(root,row,scorer,url):
    """接口：新预览→真实获取→六策略输入；只接收公开题目，不读取计分文件。"""
    from videoqa_methods.budget_contract import budget_config
    from videoqa_methods.density_selection import build_regions, refinement_targets
    from videoqa_methods.rd_selection import select_rd
    from videoqa_budget32.frames import native_frames,validate_frames
    root=Path(root);ds=row['dataset'];q=row['question_id'];dest=root/'prepared'/ds/(q+'.json')
    if dest.exists():
        saved=read_json(dest)
        for p,h in saved['exports'].items():
            if sha256(root/p)!=h:raise ValueError('Prepared image/selection changed')
        return saved
    index=read_json(root/'indices'/(row['video_id']+'.json'));video=index['video'];coarse=index['coarse']
    begin=time.perf_counter();previews=read_previews(root,index);read_seconds=time.perf_counter()-begin
    scores,features,score_time=encode(scorer,previews,row['question']);t=time.perf_counter()
    plan=build_regions(coarse,scores,video,budget_config(8));region_seconds=time.perf_counter()-t
    ranked=sorted(range(len(coarse)),key=lambda i:(-scores[i],coarse[i]['source_pts'],i))[:8]
    t=time.perf_counter();wi=wfs_indices(scores,features);wfs_seconds=time.perf_counter()-t
    chosen={'BASE-Uniform':native_frames(np.load(index['pts_path'],allow_pickle=False),video,8),
            'BASE-BLIP-TopK':sorted([coarse[i] for i in ranked],key=lambda f:f['source_pts']),
            'REF-WFS-SB':[coarse[i] for i in wi],'CTRL-WFS-preview':[coarse[i] for i in wi]}
    items={};exports={};method_time={};rd_trace=None
    # 步骤1每种方法独立原图缓存，避免共享执行时把别人的原片访问视为免费。
    for method in METHODS:
        t=time.perf_counter();selection_seconds=wfs_seconds if 'WFS' in method else 0.
        client=AccessClient(url,row['video_id'],ds+'/'+q+'/'+method);fine_time=dict(frames=0,itm_frames=0,preprocess_seconds=0.,forward_seconds=0.)
        try:
            if method in ('CTRL-TopK-local','RD-1.2-exp12'):
                if method=='CTRL-TopK-local':
                    targets=[(Fraction(str(coarse[i]['requested_seconds']))+Fraction(1,2),Fraction(0),Fraction(video['duration_fraction']),0) for i in ranked]
                else:
                    targets=[(target,Fraction(r['left_fraction']),Fraction(r['right_fraction']),r['region_id']) for r in plan['regions'] if r['refine'] for target in refinement_targets(r,video,budget_config(8))]
                added=targets_to_frames(targets,index,[f['source_pts'] for f in coarse])
                for i,f in enumerate(added):f['candidate_index']=len(coarse)+i
                full=client.fetch(added) if added else [];fine=[];preview_begin=time.perf_counter()
                for im in full:
                    with Image.open(io.BytesIO(preview_bytes(im))) as p:fine.append(p.convert('RGB'))
                    im.close()
                fine_preview_seconds=time.perf_counter()-preview_begin
                try:
                    fine_scores,vf,fine_time=encode(scorer,fine,None if method=='RD-1.2-exp12' else row['question']) if fine else ([],np.empty((0,768),np.float32),fine_time)
                    if method=='CTRL-TopK-local':
                        numerical_begin=time.perf_counter()
                        allc=coarse+added;values=scores+fine_scores
                        ids=sorted(range(len(allc)),key=lambda i:(-values[i],allc[i]['source_pts'],i))[:8]
                        chosen[method]=sorted([allc[i] for i in ids],key=lambda f:f['source_pts'])
                        selection_seconds+=time.perf_counter()-numerical_begin
                    else:
                        pool=dict(video=video,candidates=coarse+added,initial_count=len(coarse),initial_scores=scores,plan=plan,new_frames_have_itm_scores=False)
                        numerical_begin=time.perf_counter()
                        ids,rd_trace=select_rd(pool,np.concatenate([features,vf]),8,'RD-1.2-exp12',trace_mode='full')
                        selection_seconds+=region_seconds+time.perf_counter()-numerical_begin
                        chosen[method]=[pool['candidates'][i] for i in ids]
                        for rel,value in [(f'pools/{ds}/{q}.json',pool),(f'traces/{ds}/{q}.json',rd_trace)]:
                            freeze(root/rel,value);exports[rel]=sha256(root/rel)
                        rel=f'features/{ds}/{q}.npy';save_array(root/rel,np.concatenate([features,vf]));exports[rel]=sha256(root/rel)
                finally:
                    for im in fine:im.close()
            selection=dict(video=video,selected_frames=chosen[method]);validate_frames(selection,8)
            if method=='CTRL-WFS-preview':images=[previews[i].copy() for i in wi]
            else:images=client.fetch(selection['selected_frames'])
            records=[]
            try:
                for slot,(f,im) in enumerate(zip(selection['selected_frames'],images)):
                    rel=f'input_rgb/{ds}/{q}/{method}/{slot}.png';p=root/rel;p.parent.mkdir(parents=True,exist_ok=True);im.save(p)
                    exports[rel]=sha256(p);records.append(dict(f,image=rel,rgb_sha256=rgb_hash(im)))
            finally:
                for im in images:im.close()
            items[method]=dict(selection=selection,frames=records,requests=client.requests,
                extra_visual=fine_time,preparation_seconds=time.perf_counter()-t,
                selection_compute_seconds=selection_seconds,
                fine_preview_seconds=fine_preview_seconds if method in ('CTRL-TopK-local','RD-1.2-exp12') else 0.,
                preview_scoring_charged=method!='BASE-Uniform',wfs_selection_seconds=wfs_seconds if 'WFS' in method else 0.)
        finally:client.close()
    for im in previews:im.close()
    rel=f'coarse_scores/{ds}/{q}.json';freeze(root/rel,dict(scores=scores,source_frames=coarse,preview_read_seconds=read_seconds,score_timing=score_time));exports[rel]=sha256(root/rel)
    result=dict(dataset=ds,question_id=q,items=items,exports=exports,preview_scoring=score_time,preview_read_seconds=read_seconds,
        shared_asset_accounting='Common score computed once, charged independently to every scoring method; not independently timed E2E',utc=utc())
    freeze(dest,result);return result
