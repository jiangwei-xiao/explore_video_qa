"""独立证据审计：冻结输入、精确PTS视图、人工标注及保守汇总。"""
import argparse
import collections
import hashlib
import html
import json
import os
from pathlib import Path
import time

METHODS = ('uniform', 'topk', 'A', 'B', 'C', 'D')
PILOT = ('004-3','260-2','383-2','538-1','012-3','844-1','395-2','264-2','872-2','491-2')
LABELS = ('direct','context','topic','irrelevant','uncertain')
SUFFICIENCY = ('sufficient','partial','missing','uncertain')


def read(path):
    """读取UTF-8 JSON，不修改输入文件。"""
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    """分块计算资产SHA256，避免一次读取大视频。"""
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, data):
    """原子保存派生产物；历史结果路径从不传入本接口。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def selected(record):
    """兼容历史基线与方法结果结构，返回已冻结的16个源帧。"""
    return record.get('pool', record.get('selection'))['selected_frames']


def prepare(source, out, requirements):
    """建立50题审计清单；验证配对、帧预算及结果身份，不重算问答。"""
    source, out = Path(source).resolve(), Path(out).resolve()
    if out == source or source in out.parents or out in source.parents:
        raise ValueError('审计目录不能位于历史实验目录内部')
    # 步骤1：只从明确的六组结果读取，保存逐文件哈希，拒绝缺失和额外题号。
    records = {}
    hashes = {}
    for method in METHODS:
        root = source / ('baseline_reference/results' if method in METHODS[:2] else 'results') / method
        records[method] = {}
        for path in sorted(root.glob('*.json')):
            record = read(path)
            qid = record['question_id']
            if qid in records[method] or record['status'] != 'completed':
                raise ValueError('重复题号或结果未完成')
            records[method][qid] = record
            hashes[str(path)] = digest(path)
        if len(records[method]) != 50:
            raise ValueError(f'{method}不是50题')
    ids = sorted(records['A'], key=lambda q: records['A'][q]['order_index'])
    if any(set(records[m]) != set(ids) for m in METHODS):
        raise ValueError('六组题目不配对')
    if set(requirements) != set(ids):
        raise ValueError('证据需求与冻结50题不一致')
    # 步骤2：生成不带方法输出的证据清单；回答放在单独揭盲文件中。
    questions, outcomes = [], {}
    for qid in ids:
        base = records['A'][qid]
        frames, groups = {}, {}
        alias_order = sorted(METHODS, key=lambda m: hashlib.sha256((qid + m + '2027').encode()).hexdigest())
        outcomes[qid] = {}
        for method in METHODS:
            record = records[method][qid]
            for key in ('question','options','reference_answer','video_id','source_video_sha256'):
                if record[key] != base[key]:
                    raise ValueError(f'{qid}的{key}不一致')
            chosen = selected(record)
            pts = [r['source_pts'] for r in chosen]
            if len(pts) != 16 or len(set(pts)) != 16 or pts != sorted(pts):
                raise ValueError('输入不满足16个有序唯一帧')
            if record['answer']['visual_tokens'] != 3360:
                raise ValueError('视觉Token不符')
            alias = 'G' + str(alias_order.index(method) + 1)
            groups[alias] = pts
            outcomes[qid][alias] = dict(method=method, prediction=record['answer']['parsed_answer'],
                                       reference=record['reference_answer'], correct=record['correct'])
            for r in chosen:
                if r['source_pts'] in frames and frames[r['source_pts']]['source_frame_index'] != r['source_frame_index']:
                    raise ValueError('同PTS源帧编号冲突')
                frames[r['source_pts']] = r
        question = {k: base[k] for k in ('question_id','question','options','stratum','order_index','video_id','source_video_sha256')}
        question.update(video=base['pool']['video'], frames=sorted(frames.values(),key=lambda r:r['source_pts']),
                        groups=groups, pilot=qid in PILOT, requirements=requirements[qid],
                        review_status='pending', prior_outcome_exposure=True)
        questions.append(question)
    protocol = dict(schema='videoqa-evidence-audit-v1', source=str(source), result_hashes=hashes,
                    final_qa_calls=0, scope_calls=0, scoring_calls=0, pilot=list(PILOT),
                    labels=list(LABELS), sufficiency=list(SUFFICIENCY),
                    warning='事后人工审核；匿名呈现不代表审核者未接触历史结果。pending不是uncertain。')
    # 步骤3：已有审计只能身份一致续用；不覆盖任何已有人工标注。
    if (out/'protocol.json').exists() and read(out/'protocol.json') != protocol:
        raise ValueError('审计身份改变，请使用新目录')
    if (out/'questions.json').exists() and read(out/'questions.json') != questions:
        raise ValueError('题面需求/源帧身份发生变化；修订应单独留痕，不能用prepare覆盖')
    write(out/'protocol.json',protocol)
    write(out/'questions.json',questions)
    write(out/'outcomes.json',outcomes)
    write(out/'requirements_identity.json',dict(sha256=hashlib.sha256(json.dumps(requirements,ensure_ascii=False,sort_keys=True).encode()).hexdigest(),
                                              warning='题面初标修订需保存版本，不覆盖原始标注'))
    for q in questions:
        target = out/'annotations'/f'{q["question_id"]}.json'
        if not target.exists():
            write(target, dict(question_id=q['question_id'], reviewer='assistant',status='pending',
                               frames={}, groups={}, source_checks=[], notes=[]))
    return questions


def render(out, processor_path):
    """仅CPU解码与官方图像预处理；保存精确源帧及反归一化输入视图。"""
    from PIL import Image, ImageDraw, ImageOps
    import numpy as np
    import av
    from transformers import SiglipImageProcessor
    out = Path(out)
    processor = SiglipImageProcessor.from_pretrained(processor_path, local_files_only=True)
    write(out/'preprocessor.json', processor.to_dict())
    # 步骤1：前十题优先，然后冻结题序；逐题完成标志保证后台中断可以续跑。
    questions = sorted(read(out/'questions.json'),key=lambda q:(not q['pilot'], q['order_index']))
    for q in questions:
        qid = q['question_id']; root = out/'frames'/qid
        if (root/'complete.json').exists():
            continue
        root.mkdir(parents=True,exist_ok=True)
        if digest(q['video']['path']) != q['source_video_sha256']:
            raise ValueError('原视频哈希不符')
        thumbnails = []
        # 步骤2：按已知PTS回退到关键帧后向前解码，不以平均FPS推算。
        for index, row in enumerate(q['frames']):
            pts = row['source_pts']
            with av.open(q['video']['path']) as container:
                stream = container.streams.video[0]
                stream.codec_context.thread_count = 2
                container.seek(pts, stream=stream, backward=True)
                original = None
                for frame in container.decode(stream):
                    if frame.pts == pts:
                        original = frame.to_image(); break
                    if frame.pts is not None and frame.pts > pts:
                        break
                if original is None:
                    raise ValueError(f'{qid}无法恢复精确PTS {pts}')
            original.save(root/f'{pts}.original.png')
            pixels = processor(images=original, return_tensors='np')['pixel_values'][0]
            if pixels.shape != (3,384,384):
                raise ValueError('官方预处理输出尺寸不符')
            rgb = pixels.transpose(1,2,0) * np.asarray(processor.image_std) + np.asarray(processor.image_mean)
            input_view = Image.fromarray(np.rint(np.clip(rgb,0,1)*255).astype('uint8'))
            input_view.save(root/f'{pts}.input.png')
            thumb = Image.new('RGB',(320,210),'white')
            small = ImageOps.contain(original,(320,180)); thumb.paste(small,((320-small.width)//2,0))
            ImageDraw.Draw(thumb).text((3,182),f'{index+1} | {row["timestamp_seconds"]:.3f}s | PTS {pts}',fill='black')
            thumbnails.append(thumb)
        # 步骤3：24帧分页图板仅作索引；文字/细节需点击原图与384输入检查。
        for start in range(0,len(thumbnails),24):
            page = Image.new('RGB',(1280,210*6),'white')
            for i,thumb in enumerate(thumbnails[start:start+24]):
                page.paste(thumb,((i%4)*320,(i//4)*210))
            page.save(root/f'union_{start//24+1:02d}.jpg',quality=94)
        write(root/'complete.json',dict(count=len(thumbnails), video_sha256=q['source_video_sha256'],
                                       source_frame_index_status='沿用历史独立逐帧编号审计，本次核验PTS与视频哈希'))
        write(out/'render_status.json',dict(last_question=qid, completed=sum((out/'frames'/x['question_id']/'complete.json').exists() for x in questions),total=50))
        print(f'已生成 {qid}: {len(thumbnails)} 张唯一源帧',flush=True)


def validate_annotation(q, annotation):
    """校验人工标注边界；待审核条目不转成任何已审核标签。"""
    if annotation['question_id'] != q['question_id']:
        raise ValueError('题号不符')
    known = {str(r['source_pts']) for r in q['frames']}
    for pts, frame in annotation.get('frames',{}).items():
        if pts not in known or frame.get('label') not in LABELS or not frame.get('reason','').strip():
            raise ValueError('帧标签/理由/PTS不合法')
    for alias, group in annotation.get('groups',{}).items():
        if alias not in q['groups'] or group.get('sufficiency') not in SUFFICIENCY or not group.get('reason','').strip():
            raise ValueError('组级证据结论不合法')
        for pts, override in group.get('overrides',{}).items():
            if int(pts) not in q['groups'][alias] or override.get('label') not in LABELS or not override.get('reason','').strip():
                raise ValueError('组内上下文标签不合法')
        for pts, detail in group.get('redundancy',{}).items():
            if int(pts) not in q['groups'][alias] or detail.get('duplicate') not in (True,False,None) or not detail.get('reason','').strip():
                raise ValueError('组内重复标注不合法')
    if annotation.get('status') == 'initial_reviewed':
        if set(annotation.get('frames',{})) != known or set(annotation.get('groups',{})) != set(q['groups']):
            raise ValueError('完成态缺少帧或组')
        for alias, group in annotation['groups'].items():
            if set(group.get('redundancy',{})) != {str(p) for p in q['groups'][alias]}:
                raise ValueError('完成态缺少组内重复/新增信息判断')


def summarize(out):
    """分别统计问答事实与人工证据覆盖率；未审题禁止混入语义结论。"""
    out = Path(out); questions = read(out/'questions.json'); outcomes=read(out/'outcomes.json')
    states=collections.Counter(); patterns=collections.Counter(); scope={}; rows=[]
    # 步骤1：全部问答均可统计；范围标签仍保留人工初步性质。
    for q in questions:
        qid=q['question_id']; a=read(out/'annotations'/f'{qid}.json'); validate_annotation(q,a)
        states[a['status']]+=1
        flags=[x['correct'] for x in outcomes[qid].values()]
        patterns['all_correct' if all(flags) else 'all_wrong' if not any(flags) else 'discordant']+=1
        bucket=q['requirements']['scope']
        scope.setdefault(bucket,{m:dict(n=0,correct=0) for m in METHODS})
        for alias, pts in q['groups'].items():
            result=outcomes[qid][alias]; method=result['method']; scope[bucket][method]['n']+=1; scope[bucket][method]['correct']+=int(result['correct'])
            labels=collections.Counter(); group=a.get('groups',{}).get(alias,{})
            for p in pts:
                record=group.get('overrides',{}).get(str(p),a.get('frames',{}).get(str(p),{}))
                labels[record.get('label','pending')]+=1
            rows.append(dict(question_id=qid,method=method,alias=alias,correct=result['correct'],counts=dict(labels),
                             sufficiency=group.get('sufficiency','pending'),review_status=a['status']))
    # 步骤2：统计不掩盖缺项，用户审核未到齐时保持等待校准状态。
    summary=dict(questions=50,qa_results=300,frame_slots=4800,unique_frames=sum(len(q['frames']) for q in questions),
                 patterns=dict(patterns),annotation_states=dict(states),scope_accuracy_provisional=scope,
                 final_qa_calls=0,scope_calls=0,scoring_calls=0,
                 status='awaiting_user_calibration' if states['initial_reviewed'] < 50 else 'initial_review_complete_awaiting_user',
                 scope_warning='题面人工初标，不等于已核实原片证据范围；不可直接宣称局部/全局能力差异。')
    write(out/'summary.json',summary);write(out/'per_group.json',rows)
    return summary


def pages(out):
    """生成离线匿名复核页；浏览器导出JSON，不修改助手初审和历史预测。"""
    out=Path(out); template=(Path(__file__).parent/'review.html').read_text(encoding='utf-8')
    protocol_hash=digest(out/'protocol.json'); links=[]
    for q in read(out/'questions.json'):
        a=read(out/'annotations'/f'{q["question_id"]}.json');validate_annotation(q,a)
        context_pages=[str(p.relative_to(out)) for p in sorted((out/'context'/q['question_id']).glob('*/context_*.jpg'))]
        scope_reviews=read(out/'scope_reviews.json').get('questions',{}) if (out/'scope_reviews.json').exists() else {}
        data=dict(question=q,initial=a,initial_annotation_sha256=digest(out/'annotations'/f'{q["question_id"]}.json'),
                  protocol_sha256=protocol_hash,context_pages=context_pages,scope_review=scope_reviews.get(q['question_id']))
        # 步骤1：页面载荷不含方法映射、预测、标准答案；防止文本关闭script标签。
        payload=json.dumps(data,ensure_ascii=False).replace('<','\\u003c')
        target=out/'review'/f'{q["question_id"]}.html';target.parent.mkdir(exist_ok=True)
        target.write_text(template.replace('__AUDIT_DATA__',payload),encoding='utf-8')
        links.append(f'<li><a href="review/{q["question_id"]}.html">{q["question_id"]}</a> {"首批复核" if q["pilot"] else ""} — {a["status"]} — {html.escape(q["question"])}</li>')
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>Video-MME 50题证据审计</h1><p>事后审核；未审核不计为不确定。六组历史答案未重新生成。页面不包含方法映射或答案；导出复核JSON后交回。</p><ul>'+''.join(links)+'</ul>',encoding='utf-8')


def import_review(out, path):
    """验证并追加用户复核事件，保留助手初审；不自动将分歧裁决为真值。"""
    out=Path(out); data=read(path)
    if data.get('protocol_sha256') != digest(out/'protocol.json'):
        raise ValueError('复核不属于当前审计')
    q=next(q for q in read(out/'questions.json') if q['question_id']==data['question_id'])
    if set(data['groups']) != set(q['groups']):
        raise ValueError('复核分组缺失')
    if data.get('scope','') not in ('','single_moment','local_sequence','cross_time','whole_video','uncertain'):
        raise ValueError('非法范围标签')
    # 步骤1：逐组逐帧检查；空值是未作判断，不填成同意初审。
    agree=compared=0
    # 步骤1a：绑定用户所见初审版本，不能用后来修订的标签计算历史一致率。
    # 旧页面没有版本信息时仍接收用户意见，但不冒充可比较的双人标注。
    initial = {}
    initial_sha = data.get('initial_annotation_sha256')
    comparison_status = 'unversioned_review'
    if initial_sha:
        if not isinstance(initial_sha,str) or len(initial_sha)!=64 or any(c not in '0123456789abcdef' for c in initial_sha):
            raise ValueError('非法初审版本哈希')
        candidates=[out/'annotations'/f'{q["question_id"]}.json',out/'annotation_history'/q['question_id']/f'{initial_sha}.json']
        version=next((p for p in candidates if p.exists() and digest(p)==initial_sha),None)
        comparison_status='matched_initial_version' if version else 'initial_version_unavailable'
        if version:initial=read(version)
    for alias, group in data['groups'].items():
        if set(group['frames']) != {str(p) for p in q['groups'][alias]}:
            raise ValueError('复核帧不匹配')
        if group['sufficiency'] not in ('',)+SUFFICIENCY:
            raise ValueError('非法证据完整性')
        for pts, item in group['frames'].items():
            if item['label'] not in ('',)+LABELS:
                raise ValueError('非法标签')
            if item.get('duplicate','') not in ('','yes','no','uncertain'):
                raise ValueError('非法重复性标签')
            prior=initial.get('groups',{}).get(alias,{}).get('overrides',{}).get(pts,initial.get('frames',{}).get(pts,{}))
            if item['label'] and prior.get('label'):
                compared+=1;agree+=int(item['label']==prior['label'])
    # 步骤2：内容寻址实现幂等导入，原始审核与一致率分开保存。
    fingerprint=digest(path);target=out/'reviews'/f'{q["question_id"]}.{fingerprint[:16]}.json'
    if not target.exists():write(target,data)
    write(out/'reviews'/f'{q["question_id"]}.{fingerprint[:16]}.agreement.json',
          dict(compared_slots=compared,agree_slots=agree,agreement=agree/compared if compared else None,
               initial_annotation_sha256=initial_sha,comparison_status=comparison_status,
               warning='同一源帧重复出现的槽位非独立样本；显示初审后提交不是独立盲审。'))


def main():
    """命令行入口：分离准备、后台渲染、汇总和用户复核导入。"""
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','render','report','import-review'])
    p.add_argument('--out',required=True);p.add_argument('--source');p.add_argument('--requirements');p.add_argument('--review')
    p.add_argument('--processor',default='/home/models/google/siglip-so400m-patch14-384')
    args=p.parse_args()
    if args.action=='prepare':prepare(args.source,args.out,read(args.requirements));pages(args.out);print(summarize(args.out))
    elif args.action=='render':
        import fcntl
        with (Path(args.out)/'render.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            render(args.out,args.processor);pages(args.out);summarize(args.out)
    elif args.action=='report':
        pages(args.out);print(summarize(args.out))
        from .report import export
        print(export(args.out))
    else:import_review(args.out,args.review)
