"""独立长视频试点：先冻结资格/划分，再允许教师反馈；不改历史RD实现。"""
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from pathlib import Path
import json
import os
import re
import shutil
import subprocess
import time
import traceback

import numpy as np

from videoqa_runtime.common import ROOT, read_json, sha256
from videoqa_full.state import durable, digest, exclusive, utc
from videoqa_set_value.data import METHODS, near_duplicates, phash_image
from videoqa_set_value.data_acquisition import stable_key
from videoqa_set_value.media import SequentialFrameProvider
from videoqa_set_value.longvideo_assets import (ANNOTATION_REVISION, VIDEO_REVISION, inventory,
    asset_state, extract_archive)

SOURCE_SCOPE='set_value_longvideo_v1'
OLD_DATA=ROOT/'outputs/set_value/set_value_learning__nextqa800__llava8__20261019_r2'
REFERENCE=ROOT/'outputs/set_value/set_value_learning__nextqa800__llava8__20261019_r1/reference_bank.json'
OLD_HEAD=ROOT/'outputs/analysis/set_value_trainability__nextqa600__20261019_r1/full_fit/state.pt'
OLD_HEAD_SHA='8f069aa8288dbe012d1e7e9b69e2e18e46126ab904dfd6d8c02f54d5f398655f'


def parse_problem(text):
    """接口：仅解析公开原题/四选项，保留原字母显示顺序；不读取答案或reasoning。

    只归一化明确的换行转义和CRLF，禁止unicode_escape破坏引号、Unicode或路径。
    """
    if not isinstance(text,str):raise ValueError('Problem must be original text')
    value=text.replace('\r\n','\n').replace('\\r\\n','\n').replace('\\n','\n').strip()
    pattern=re.compile(r'(?m)^\s*(?:\(([A-D])\)|([A-D])[.)])\s+')
    matches=list(pattern.finditer(value))
    if len(matches)!=4:raise ValueError('Exactly four unambiguous labelled option lines required')
    labels=[m.group(1) or m.group(2) for m in matches]
    if set(labels)!=set('ABCD'):raise ValueError('Repeated or missing original option letter')
    question=value[:matches[0].start()].strip()
    options=[value[m.end():matches[i+1].start() if i+1<4 else len(value)].strip() for i,m in enumerate(matches)]
    if not question or any(not s for s in options) or len(set(options))!=4:
        raise ValueError('Question/options empty or not distinct')
    return dict(question=question,options=options,option_letters=labels)


def source_id(value):
    """接口：仅接受官方相对视频路径中的单个YouTube源ID，拒绝多视频/路径歧义。"""
    if not isinstance(value,str) or not re.fullmatch(r'longvila_videos/[A-Za-z0-9_-]{11}\.mp4',value):
        raise ValueError('Ambiguous source video identity')
    return Path(value).stem


def metadata(asset_root):
    """接口：固定三个Parquet校验后读取；候选资格不读取reasoning或参考答案内容。"""
    import pyarrow.parquet as pq
    asset_root=Path(asset_root);dest=asset_root/'metadata_projection.json'
    if dest.exists():
        old=read_json(dest)
        for path,expected in old['sources'].items():
            if sha256(path)!=expected:raise ValueError('Pinned annotation changed')
        return old
    sources={};tables={}
    # 步骤1：完整SHA匹配固定Parquet，不使用已发现转义异常的原始JSONL。
    for entry in inventory()[:3]:
        state=asset_state(asset_root,entry['name'])
        if not state['complete']:raise ValueError('Pinned annotation not ready')
        path=state['path']
        if sha256(path)!=entry['sha256']:raise ValueError('Annotation SHA mismatch')
        sources[path]=entry['sha256'];tables[entry['name'].split('-')[0]]=pq.read_table(path).to_pylist()
    if len(tables['train'])!=51200:raise ValueError('Official training row denominator differs')
    excluded_ids=set();excluded_errors=[]
    for split in ('validation','test'):
        for ordinal,row in enumerate(tables[split]):
            try:
                # 官方test还含MLVU样式的数字ID/webm；保留完整基名排除，不误套train的YT格式。
                value=row['videos']
                if not isinstance(value,str) or not re.fullmatch(r'longvila_videos/[A-Za-z0-9_-]+\.(?:mp4|webm|mkv)',value):
                    raise ValueError('Unsafe held-out source identity')
                excluded_ids.add(Path(value).stem)
            except ValueError as error:excluded_errors.append(dict(split=split,ordinal=ordinal,error=str(error)))
    if excluded_errors:raise ValueError('Official held-out video identities cannot be fully isolated')
    public=[];invalid=[];private=[]
    # 步骤2：公开投影与计分标签分文件；解析/哈希/资格/划分不读取答案内容。
    for ordinal,row in enumerate(tables['train']):
        try:
            if row['data_type']!='video':raise ValueError('Not a single-video training record')
            vid=source_id(row['videos']);parsed=parse_problem(row['problem']);qid='lvr-train-'+str(row['problem_id'])
            public.append(dict(question_id=qid,official_problem_id=row['problem_id'],source_row=ordinal,video_id=vid,
                original_problem=row['problem'],task_type=row['problem_type'],**parsed))
            private.append(dict(question_id=qid,reference_answer=row['answer']))
        except ValueError as error:invalid.append(dict(source_row=ordinal,problem_id=row.get('problem_id'),reason=str(error)))
    if len({r['question_id'] for r in public})!=len(public):raise ValueError('Ambiguous official problem identity')
    durable(asset_root/'scoring_labels.json',dict(rows=private,source_sha256=sources))
    result=dict(annotation_revision=ANNOTATION_REVISION,video_revision=VIDEO_REVISION,sources=sources,
        official_train_rows=51200,rows=public,invalid=invalid,heldout_source_ids=sorted(excluded_ids),
        private_labels_path=str(asset_root/'scoring_labels.json'),private_labels_sha256=sha256(asset_root/'scoring_labels.json'),
        qualification_uses_reference_answers=False,reasoning_exported=False,utc=utc())
    durable(dest,result);return result


def known_content():
    """接口：既有源ID/SHA/感知指纹只供数据隔离，禁止读取历史问答对错。"""
    reference=read_json(REFERENCE);bank=list(reference['videos']);sources={str(REFERENCE):sha256(REFERENCE)}
    nextqa=read_json(OLD_DATA/'manifest.json');sources[str(OLD_DATA/'manifest.json')]=sha256(OLD_DATA/'manifest.json')
    by_sha={r['video_sha256'] for r in bank};known_sha=set(by_sha);ids={r['source_id'].split(':',1)[1] for r in bank}
    for row in nextqa['rows']:
        ids.add(str(row['video_id']));h=row['video_sha256']
        if h in by_sha:continue
        checks=list((OLD_DATA/'candidate_checks').glob(str(row['video_id'])+'.json'))
        if not checks:
            # 父准备可复用r1，内容资格记录不可由后续教师输出补写。
            checks=list((REFERENCE.parent/'candidate_checks').glob(str(row['video_id'])+'.json'))
        if not checks:raise ValueError('Missing existing NExT-QA visual identity')
        cp=checks[0];record=read_json(cp)
        if not record.get('eligible') or record['video_sha256']!=h:raise ValueError('Existing source identity differs')
        sources[str(cp)]=sha256(cp);bank.append(dict(source_id='nextqa:'+str(row['video_id']),video_sha256=h,
            fingerprint=record['fingerprint']));by_sha.add(h)
    # 步骤1：既有资产包括未入原800清单的候选；坏流也保留已知源ID/SHA，但不伪造指纹。
    for cp in sorted((REFERENCE.parent/'candidate_checks').glob('*.json')):
        record=read_json(cp);h=record['video_sha256'];vid=str(record['video_id'])
        sources[str(cp)]=sha256(cp);known_sha.add(h);ids.add(vid)
        if record.get('eligible') and h not in by_sha:
            bank.append(dict(source_id='nextqa:'+vid,video_sha256=h,fingerprint=record['fingerprint']));by_sha.add(h)
    vm=ROOT/'data/manifests/video_mme_full_2700.json';sources[str(vm)]=sha256(vm)
    ids.update(r['video_id'] for r in read_json(vm)['rows'])
    return bank,ids,sources,known_sha


def duration_check(job):
    """接口：容器时长用于初筛；合格者仍须完整解码并以真实流PTS验收180–1800秒。"""
    root,record=Path(job[0]),job[1];path=root/'duration_checks'/(record['video_id']+'.json')
    if path.exists():return read_json(path)
    command=['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=duration,time_base,start_pts',
        '-of','json',record['path']]
    try:
        value=json.loads(subprocess.check_output(command,text=True,timeout=90))['streams'][0]
        duration=float(value['duration']);eligible=180.<=duration<=1800.
        result=dict(**record,duration_seconds=duration,eligible=eligible,reason=None if eligible else 'duration_outside_180_1800')
    except (ValueError,KeyError,IndexError,subprocess.SubprocessError) as error:
        result=dict(**record,eligible=False,reason='container_error:'+type(error).__name__)
    durable(path,result);return result


def index_fingerprint(job):
    """接口：完整真实PTS索引及8帧指纹，源号为解码序号；不使用平均FPS或问题答案。"""
    import av
    from videoqa_mlvu.data import index_one
    from videoqa_budget32.frames import native_frames
    root,record=Path(job[0]),job[1];cp=root/'identity_checks'/(record['video_id']+'.json')
    if cp.exists():
        old=read_json(cp)
        if sha256(record['path'])!=old['sha256']:raise ValueError('Candidate source bytes changed')
        return old
    # 步骤1：冻结清单前的完整解码资格独立于教师；失败记录原原因，不重编码/改范围修复。
    try:
        index_one((str(root/'source_indices'),record['sha256'],record['path']))
        ip=root/'source_indices/indices'/(record['sha256']+'.json');index=read_json(ip);video=index['video']
        if not 180.<=video['duration_seconds']<=1800.:raise ValueError('Container/decoded duration qualification differs')
        pts=np.load(index['pts_path'],allow_pickle=False)
        ims,rgb=SequentialFrameProvider(dict(video=video,selected_frames=native_frames(pts,video,8))).decode()
        try:fp=dict(phash=[phash_image(im) for im in ims],rgb_sha256=rgb)
        finally:
            for im in ims:im.close()
        result=dict(**record,index_path=str(ip),index_sha256=sha256(ip),fingerprint=fp)
    except (av.error.FFmpegError,ValueError,RuntimeError) as error:
        result=dict(record,eligible=False,reason='full_decode_qualification:'+repr(error))
    durable(cp,result);return result


def choose_split(groups, questions):
    """接口：完整合格内容组按固定视频哈希划分800/400，再按独立问题哈希取一道题。"""
    ordered=sorted(groups,key=lambda r:stable_key(SOURCE_SCOPE+'/video',r['video_id']))
    if len(ordered)<1200:raise ValueError('Fixed eight shards contain fewer than 1200 eligible independent videos')
    rows=[]
    for ordinal,record in enumerate(ordered[:1200]):
        vid=record['video_id'];options=questions[vid]
        chosen=min(options,key=lambda r:stable_key(SOURCE_SCOPE+'/question',r['official_problem_id']))
        allowed=('question_id','official_problem_id','source_row','video_id','original_problem','task_type',
            'question','options','option_letters')
        public={k:chosen[k] for k in allowed if k in chosen}
        rows.append(dict(public,ordinal=ordinal,split='train' if ordinal<800 else 'val',pilot=ordinal<200,
            video_path=record['path'],video_sha256=record['sha256'],index_path=record['index_path'],
            duration_seconds=record['duration_seconds'],content_group=record['sha256']))
    if len({r['video_id'] for r in rows})!=1200 or len({r['content_group'] for r in rows})!=1200:
        raise ValueError('Content crossed the train/validation split')
    return rows


def prepare(root, asset_root):
    """接口：固定分片全到齐后按身份资格冻结1200题；不足直接结束，不扩来源或补样。"""
    root=Path(root);asset_root=Path(asset_root);root.mkdir(parents=True,exist_ok=True)
    if (root/'manifest.json').exists():
        old=read_json(root/'manifest.json')
        for path,expected in old['identity_sources'].items():
            if sha256(path)!=expected:raise ValueError('Frozen qualification identity changed')
        if len(old['rows'])!=1200:raise ValueError('Frozen denominator differs')
        return old
    with exclusive(root/'asset_qualification.lock'):
        free=shutil.disk_usage(asset_root).free
        if free<max(600*10**9,1.5*sum(e['bytes'] for e in inventory()[3:])):
            raise ValueError('Less than 600GB or 1.5x estimated source headroom')
        meta=metadata(asset_root);questions=defaultdict(list)
        for row in meta['rows']:questions[row['video_id']].append(row)
        heldout=set(meta['heldout_source_ids']);bank,known_ids,sources,known_sha=known_content()
        sources.update(meta['sources']);sources[str(asset_root/'metadata_projection.json')]=sha256(asset_root/'metadata_projection.json')
        sources[str(asset_root/'scoring_labels.json')]=meta['private_labels_sha256']
        records=[]
        # 步骤1：八个固定归档都验收完才形成完整资格总体，绝不按先到顺序抽样。
        for entry in inventory()[3:]:
            extracted=extract_archive(asset_root,entry['name'],set(questions))
            records.extend(extracted['members'])
            durable(root/'heartbeat.json',dict(stage='extract_fixed_archives',archives_completed=len(list((asset_root/'extraction').glob('*.json'))),
                videos=len(records),pid=os.getpid(),utc=utc()))
        if len({r['video_id'] for r in records})!=len(records):raise ValueError('Ambiguous cross-shard source identities')
        rejected=[];initial=[]
        for record in records:
            reason='official_validation_or_test_identity' if record['video_id'] in heldout else (
                'known_source_id_or_content' if record['video_id'] in known_ids or record['sha256'] in known_sha else None)
            if reason:rejected.append(dict(video_id=record['video_id'],reason=reason))
            else:initial.append(record)
        with ProcessPoolExecutor(max_workers=4) as pool:
            checked=list(pool.map(duration_check,[(str(root),r) for r in initial]))
        candidates=[r for r in checked if r['eligible']]
        rejected.extend(dict(video_id=r['video_id'],reason=r['reason']) for r in checked if not r['eligible'])
        # 步骤2：先做数量上界检查；不够1200时不用完整解码/BLIP消耗掩盖来源不足。
        summary=dict(extracted_train_videos=len(records),duration_eligible=len(candidates),rejections=rejected,
            fixed_shards=[e['name'] for e in inventory()[3:]],sources=sources,model_calls=0,utc=utc())
        if len(candidates)<1200:
            result=dict(summary,status='stopped_asset_shortfall',shortfall=1200-len(candidates),learning_validation_completed=False)
            durable(root/'asset_report.json',result);durable(root/'STOP.json',dict(reason='fixed_asset_population_shortfall',utc=utc()))
            return result
        checked=[]
        with ProcessPoolExecutor(max_workers=4) as pool:
            for record in pool.map(index_fingerprint,[(str(root),r) for r in candidates]):
                checked.append(record);durable(root/'heartbeat.json',dict(stage='full_pts_and_fingerprint',completed=len(checked),total=len(candidates),pid=os.getpid(),utc=utc()))
        rejected.extend(dict(video_id=r['video_id'],reason=r['reason']) for r in checked if not r['eligible'])
        checked=[r for r in checked if r['eligible']]
        # 步骤3：建立传递性内容组；并查集避免A≈B≈C被顺序去重拆进不同划分。
        parent=list(range(len(checked)))
        def find(i):
            while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
            return i
        accepted=[];within=[];by_sha={}
        for i,record in enumerate(checked):
            duplicates=near_duplicates(record['fingerprint']['phash'],bank)
            if duplicates:
                rejected.append(dict(video_id=record['video_id'],reason='known_near_duplicate',matches=duplicates));continue
            hits=near_duplicates(record['fingerprint']['phash'],within)
            links=[int(v.split(':',1)[1]) for v in hits]
            if record['sha256'] in by_sha:links.append(by_sha[record['sha256']])
            for other in links:parent[find(i)]=find(other)
            by_sha[record['sha256']]=i;accepted.append(i)
            within.append(dict(source_id='candidate:'+str(i),fingerprint=record['fingerprint']))
        components=defaultdict(list)
        for i in accepted:components[find(i)].append(i)
        groups=[min((checked[i] for i in items),key=lambda r:stable_key(SOURCE_SCOPE+'/video',r['video_id'])) for items in components.values()]
        summary.update(independent_content_groups=len(groups),rejections=rejected,
            content_groups=[[checked[i]['video_id'] for i in items] for items in components.values()])
        if len(groups)<1200:
            result=dict(summary,status='stopped_asset_shortfall',shortfall=1200-len(groups),learning_validation_completed=False)
            durable(root/'asset_report.json',result);durable(root/'STOP.json',dict(reason='independent_content_shortfall',utc=utc()));return result
        rows=choose_split(groups,questions)
        value=dict(rows=rows,identity_sources=sources,annotation_revision=ANNOTATION_REVISION,video_revision=VIDEO_REVISION,
            train_videos=800,val_videos=400,pilot_videos=200,seed=2027,selection_reads_answers=False,
            overlap_limits=['Eight uniform fingerprints cannot exclude every partial overlap','Teacher pretraining overlap unknown'],utc=utc())
        durable(root/'asset_report.json',dict(summary,status='qualification_passed'))
        durable(root/'manifest.json',value);return value


def worker(root,asset_root,phase,gpu):
    """接口：工作进程仅负责指定阶段；实际实现与教师/训练环境隔离。"""
    from videoqa_set_value.longvideo_runtime import worker as execute
    return execute(Path(root),phase,gpu)


def run(root,asset_root):
    """接口：后台等待固定资产→资格→先导→条件性训练/验证；未知状态绝不重发生成。"""
    root=Path(root);asset_root=Path(asset_root);root.mkdir(parents=True,exist_ok=True)
    with exclusive(root/'coordinator.lock'):
        from videoqa_set_value.longvideo_runtime import freeze_sources
        freeze_sources(root)
        if (root/'PAUSE.json').exists():raise RuntimeError('Paused trial requires inspection; no automatic restart')
        if (root/'STOP.json').exists() and (root/'asset_report.json').exists() and not (root/'manifest.json').exists():
            return read_json(root/'asset_report.json')
        durable(root/'coordinator_session.json',dict(pid=os.getpid(),sid=os.getsid(0),utc=utc()))
        # 步骤1：只等待固定文件，不依赖当前SSH连接，也不在远端尝试失败的HF直连。
        began=time.monotonic();initial_bytes=None
        while not all((asset_root/'verified'/(e['name']+'.json')).exists() for e in inventory()):
            status_path=asset_root/'relay_status.json'
            if status_path.exists() and read_json(status_path)['status']=='failed':
                result=dict(status='paused_asset_relay_failure',model_calls=0,utc=utc())
                durable(root/'PAUSE.json',result);return result
            completed=[e for e in inventory() if (asset_root/'verified'/(e['name']+'.json')).exists()]
            completed_names={e['name'] for e in completed};chunk_bytes=0
            for path in (asset_root/'chunks').glob('*/*.json'):
                if path.parent.name in completed_names:continue
                try:chunk_bytes+=read_json(path)['bytes']
                except FileNotFoundError:continue  # 整包刚发布时可清理已保留在原包中的分块副本。
            received=sum(e['bytes'] for e in completed)+chunk_bytes
            if initial_bytes is None:initial_bytes=received
            elapsed=time.monotonic()-began;new_bytes=received-initial_bytes;total=sum(e['bytes'] for e in inventory())
            durable(root/'heartbeat.json',dict(stage='waiting_for_pinned_assets',files_completed=len(completed),files_total=11,
                verified_bytes=sum(e['bytes'] for e in completed),verified_chunk_bytes=chunk_bytes,total_received_bytes=received,
                total_bytes=total,eta_seconds=elapsed*(total-received)/new_bytes if new_bytes>0 else None,
                pid=os.getpid(),sid=os.getsid(0),model_calls=0,utc=utc()));time.sleep(10)
        try:
            result=prepare(root,asset_root)
            if result.get('status')=='stopped_asset_shortfall':return result
            # 步骤2：工程实现必须完整、冻结后才可进入模型阶段，资产下载不能越过先导门槛。
            from videoqa_set_value.longvideo_runtime import run as execute
            return execute(root)
        except BaseException as error:
            durable(root/'PAUSE.json',dict(stage='asset_or_trial',error=repr(error),traceback=traceback.format_exc(),utc=utc()))
            raise


def audit(root):
    """接口：完成审计只读，不重复生成、补充资产或训练第二个头。"""
    root=Path(root)
    if (root/'STOP.json').exists() and not (root/'manifest.json').exists():return read_json(root/'asset_report.json')
    from videoqa_set_value.longvideo_runtime import audit as execute
    return execute(root)
