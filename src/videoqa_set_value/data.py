"""独立学习数据清单：源级查重、完整PTS索引、固定600/200题。"""
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from pathlib import Path
import time
import fcntl
import numpy as np
from PIL import Image
from scipy.fft import dctn

from videoqa_runtime.common import ROOT,read_json,sha256
from videoqa_full.state import durable,save_array,exclusive,utc
from videoqa_budget32.frames import FrameProvider,native_frames
from videoqa_set_value.data_acquisition import official_metadata,acquire,stable_key

METHODS=('BASE-Uniform','BASE-BLIP-TopK','RD-1.2-exp07','RD-1.2-exp12')
ML=ROOT/'outputs/full_videoqa/mlvu_dev__S1__llava8_qwen32__four_methods__20261014_r1'
VM=ROOT/'outputs/full_videoqa/llava8__five_methods__videomme2700__20261005_r1'


def phash_image(image):
    """接口：固定32灰度DCT/8×8低频、去DC中位阈值形成64位感知摘要。"""
    data=np.asarray(image.convert('L').resize((32,32),Image.Resampling.LANCZOS),dtype=np.float64)
    low=dctn(data,type=2,norm='ortho')[:8,:8].reshape(-1)
    bits=low>np.median(low[1:]);bits[0]=False
    return format(sum(int(v)<<i for i,v in enumerate(bits)),'016x')


def fingerprint(selection):
    """接口：按8个真实源帧计算视觉指纹；不读取问题、标签或历史答案。"""
    ims,rgb=FrameProvider(selection).decode()
    try:return dict(phash=[phash_image(im) for im in ims],rgb_sha256=rgb,
        source_pts=[f['source_pts'] for f in selection['selected_frames']])
    finally:
        for im in ims:im.close()


def temporal_matches(a,b,distance=6):
    """接口：8×8匹配以汉明≤6为相近，LCS要求不同帧按同一时间方向对应。"""
    match=[[((int(x,16)^int(y,16)).bit_count()<=distance) for y in b] for x in a]
    dp=np.zeros((len(a)+1,len(b)+1),dtype=np.int16)
    for i in range(len(a)):
        for j in range(len(b)):
            dp[i+1,j+1]=max(dp[i,j+1],dp[i+1,j],dp[i,j]+int(match[i][j]))
    return int(dp[-1,-1])


def near_duplicates(query,bank):
    """接口：候选8帧至少6帧按序相近才标疑似重复；只作数据隔离，不过滤选帧。"""
    if not bank:return []
    a=np.asarray([int(x,16) for x in query],dtype=np.uint64)
    b=np.asarray([[int(x,16) for x in r['fingerprint']['phash']] for r in bank],dtype=np.uint64)
    xor=np.ascontiguousarray(a[:,None,None]^b[None,:,:])
    table=np.array([i.bit_count() for i in range(256)],dtype=np.uint8)
    distances=table[xor.view(np.uint8).reshape(*xor.shape,8)].sum(-1)
    possible=np.flatnonzero((distances<=6).any(axis=2).sum(axis=0)>=6)
    return [bank[int(i)]['source_id'] for i in possible if temporal_matches(query,bank[int(i)]['fingerprint']['phash'])>=6]


def _reference_one(job):
    """接口：已有均匀8帧只是视频指纹读取位置，绝不读取其回答或对错。"""
    root,path,source_id,expected_sha=Path(job[0]),Path(job[1]),job[2],job[3]
    dest=root/'reference_fingerprints'/(expected_sha+'.json')
    if dest.exists():return read_json(dest)
    selection=read_json(path)['selection'];video_path=Path(selection['video']['path'])
    actual=sha256(video_path)
    if actual!=expected_sha:raise ValueError('Frozen evaluation video bytes changed')
    result=dict(source_id=source_id,video_sha256=actual,video_path=str(video_path),
        selection_path=str(path),selection_sha256=sha256(path),fingerprint=fingerprint(selection),utc=utc())
    durable(dest,result);return result


def build_reference_bank(root):
    """接口：冻结Video-MME/MLVU源内容的重复检查银行，不读已测答案。"""
    root=Path(root)
    with exclusive(root/'reference_bank.lock'):
        if (root/'reference_bank.json').exists():return read_json(root/'reference_bank.json')
        vm_rows=read_json(ROOT/'data/manifests/video_mme_full_2700.json')['rows']
        vm_hashes=read_json(VM/'protocol.json')['video_hashes']
        ml=read_json(ML/'manifest.json');jobs=[];seen=set()
        for row in vm_rows:
            vid=row['video_id'];h=vm_hashes[vid]
            if h in seen:continue
            seen.add(h);jobs.append((str(root),str(VM/'selections/BASE-Uniform'/(row['question_id']+'.json')),'videomme:'+vid,h))
        for row in ml['rows']:
            h=row['video_id']
            if h in seen:continue
            seen.add(h);jobs.append((str(root),str(ML/'selections/llava8/BASE-Uniform'/(row['question_id']+'.json')),'mlvu:'+Path(row['video_path']).stem,h))
        bank=[]
        # 步骤1：每视频只读既有8帧位置，完整字节SHA和近重复视觉指纹分别记录。
        with ProcessPoolExecutor(max_workers=8) as pool:
            for item in pool.map(_reference_one,jobs):
                bank.append(item)
                durable(root/'data_status.json',dict(stage='reference_fingerprints',completed=len(bank),total=len(jobs),utc=utc()))
        result=dict(videos=bank,source_manifests={str(ROOT/'data/manifests/video_mme_full_2700.json'):sha256(ROOT/'data/manifests/video_mme_full_2700.json'),
            str(ML/'manifest.json'):sha256(ML/'manifest.json'),str(VM/'protocol.json'):sha256(VM/'protocol.json')},
            policy=dict(frames=8,hamming_threshold=6,minimum_ordered_matches=6),utc=utc())
        durable(root/'reference_bank.json',result);return result


def _index_candidate(job):
    """接口：新视频完整解码索引后判定粗池≥8；坏视频作为生成前资格记录。"""
    from videoqa_mlvu.data import index_one
    root,vid,path,digest=Path(job[0]),job[1],job[2],job[3]
    dest=root/'candidate_checks'/(vid+'.json')
    if dest.exists():return read_json(dest)
    try:
        index_one((str(root/'source_indices'),digest,path))
        ip=root/'source_indices/indices'/(digest+'.json');index=read_json(ip)
        pts=np.load(index['pts_path'],allow_pickle=False);video=index['video'];base=Fraction(video['time_base'])
        targets=[];t=Fraction(1,4)
        while t<Fraction(video['duration_fraction']):
            tick=(t/base+video['start_pts']);tick=(tick.numerator+tick.denominator-1)//tick.denominator
            pos=int(np.searchsorted(pts,tick))
            if pos<len(pts):targets.append(pos)
            t+=1
        if len(set(targets))<8:raise ValueError('Fewer than eight distinct coarse candidates')
        selection=dict(video=video,selected_frames=native_frames(pts,video,8))
        result=dict(video_id=vid,video_sha256=digest,path=path,eligible=True,index=str(ip),
            coarse_count=len(set(targets)),fingerprint=fingerprint(selection),utc=utc())
    except (ValueError,RuntimeError) as error:
        result=dict(video_id=vid,video_sha256=digest,path=path,eligible=False,error=repr(error),utc=utc())
    durable(dest,result);return result


def prepare_manifest(root):
    """接口：固定300训练/100验证视频，各两题；全部排除/替补在任何模型问答之前。"""
    root=Path(root)
    with exclusive(root/'manifest.lock'):
        if (root/'manifest.json').exists():return read_json(root/'manifest.json')
        metadata=official_metadata(root);references=build_reference_bank(root)['videos']
        known_sha={r['video_sha256'] for r in references};known_ids={r['source_id'].split(':',1)[1] for r in references}
        accepted={'train':[],'val':[]};rejections=[];used_sha=set();candidate_bank=[]
        counts={'train':320,'val':120}
        # 步骤1：数据下载可扩展固定候选前缀，绝不依据答案或后续模型结果补样。
        while len(accepted['train'])<300 or len(accepted['val'])<100:
            # 已有后台下载正常持锁时等待其完成；不把正常占用视作失败或重启理由。
            with (root/'acquisition.lock').open('a') as stream:
                fcntl.flock(stream,fcntl.LOCK_EX)
                fcntl.flock(stream,fcntl.LOCK_UN)
            acquire(root,counts['train'],counts['val'])
            orders=read_json(root/'candidate_video_order.json')['selected']
            jobs=[]
            for split in ('train','val'):
                for vid in orders[split]:
                    d=read_json(root/'downloads'/(vid+'.json'))
                    jobs.append((str(root),vid,d['path'],d['sha256']))
            with ProcessPoolExecutor(max_workers=8) as pool:checked=list(pool.map(_index_candidate,jobs))
            lookup={x['video_id']:x for x in checked};accepted={'train':[],'val':[]};rejections=[];used_sha=set();candidate_bank=[]
            for split,target in (('train',300),('val',100)):
                for vid in orders[split]:
                    if len(accepted[split])==target:break
                    c=lookup[vid];reason=None
                    if not c['eligible']:reason=c.get('error','index_failure')
                    elif c['video_sha256'] in known_sha or vid in known_ids:reason='existing_evaluation_source_identity'
                    elif c['video_sha256'] in used_sha:reason='duplicate_new_source_sha'
                    else:
                        duplicates=near_duplicates(c['fingerprint']['phash'],references+candidate_bank)
                        if duplicates:reason='suspected_near_duplicate:'+','.join(duplicates)
                    if reason:rejections.append(dict(split=split,video_id=vid,reason=reason));continue
                    accepted[split].append(c);used_sha.add(c['video_sha256']);candidate_bank.append(dict(
                        source_id='nextqa:'+vid,fingerprint=c['fingerprint']))
            if len(accepted['train'])<300:counts['train']+=20
            if len(accepted['val'])<100:counts['val']+=20
        # 步骤2：问题排序只按复合源身份，各视频取2道；标签仅原样保存供生成后计分。
        rows=[]
        for split in ('train','val'):
            grouped={}
            for source in metadata[split]:grouped.setdefault(source['video'],[]).append(source)
            for c in accepted[split]:
                qs=sorted(grouped[c['video_id']],key=lambda r:stable_key('question',r['video']+'/'+r['type']+'/'+r['qid']+'/'+str(r['_row_index'])))[:2]
                for source in qs:
                    q='nextqa-'+split+'-'+source['video']+'-'+source['type']+'-'+source['qid']+'-'+str(source['_row_index'])
                    rows.append(dict(question_id=q,split=split,video_id=c['video_id'],video_path=c['path'],video_sha256=c['video_sha256'],
                        index_path=c['index'],question=source['question'],options=[source['a'+str(i)] for i in range(5)],answer_index=int(source['answer']),
                        task_type=source['type'],official_qid=source['qid'],official_row=source['_row_index']))
        if len(rows)!=800 or len({r['question_id'] for r in rows})!=800:raise ValueError('800 questions do not close')
        manifest=dict(rows=rows,accepted_videos=accepted,rejections=rejections,seed=2027,
            train_questions=600,val_questions=200,methods=list(METHODS),frame_budget=8,
            metadata_identity_sha256=sha256(root/'official_identity.json'),reference_bank_sha256=sha256(root/'reference_bank.json'),
            sampling_uses_model_outcomes=False,utc=utc())
        durable(root/'manifest.json',manifest);return manifest
