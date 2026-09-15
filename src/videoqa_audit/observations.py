"""将逐页人工观察记录展开为源PTS标注；不按模型分数自动判断合理性。"""
from pathlib import Path
from .core import read,write,digest,validate_annotation


def indices(expression):
    """展开人工填写的1起始图板编号区间，拒绝重复编号。"""
    result=[]
    for piece in expression.split(','):
        if '-' in piece:
            left,right=map(int,piece.split('-'));result.extend(range(left,right+1))
        elif piece.strip():result.append(int(piece))
    if len(result)!=len(set(result)):raise ValueError('编号重复')
    return result


def apply_observations(out, specifications, revision_reason=None):
    """人工显式标签必须覆盖每张已查看图板帧；原始观察和展开标注分别保存。"""
    out=Path(out);questions={q['question_id']:q for q in read(out/'questions.json')}
    for qid,spec in specifications.items():
        q=questions[qid];labels={};units={}
        # 步骤1：所有帧的语义标签来自人工观察；缺项立即停止，绝不填默认相关/无关。
        for expression,label,reason in spec['observations']:
            for i in indices(expression):
                if i in labels:raise ValueError(f'{qid}重复标注{i}')
                labels[i]=dict(label=label,reason=reason,viewed='原始PTS图板；384输入另记核查范围')
        if set(labels)!=set(range(1,len(q['frames'])+1)):raise ValueError(f'{qid}未覆盖全部唯一帧')
        # 步骤2：重复单元由人工指定同一画面/同一证据，不用时间距离或CLS阈值代替语义。
        for expression,name in spec.get('same_evidence_units',[]):
            for i in indices(expression):
                if i in units:raise ValueError('证据单元重叠')
                units[i]=name
        frames={str(row['source_pts']):dict(labels[i],evidence_unit=units.get(i,f'unique_{i}')) for i,row in enumerate(q['frames'],1)}
        groups={}
        for alias,pts in q['groups'].items():
            seen={};redundancy={}
            for p in pts:
                f=frames[str(p)];unit=f['evidence_unit'];previous=seen.get(unit)
                uncertain=f['label']=='uncertain'
                redundancy[str(p)]=dict(duplicate=None if uncertain else previous is not None,
                    reason='无法可靠判断新增信息' if uncertain else f'与本组PTS {previous} 属于人工确认的同一证据单元' if previous else '本组首次出现该人工证据单元；不意味着足以回答',
                    new_information=None if uncertain else previous is None and f['label'] in ('direct','context'))
                seen[unit]=p
            groups[alias]=dict(sufficiency=spec['groups'][alias][0],reason=spec['groups'][alias][1],overrides={},redundancy=redundancy)
        annotation=dict(question_id=qid,reviewer='assistant',status='initial_reviewed',frames=frames,groups=groups,
                        notes=spec['notes'],source_checks=spec.get('source_checks',[]),
                        model_input_visibility='关键帧384输入核查记录见input_visibility；其余逐帧384可见性仍待复核',
                        input_visibility=spec.get('input_visibility',{}), prior_outcome_exposure=True,
                        scope_status='题面初标；原片仅核查已记录窗口，并非全片完整证据标注')
        validate_annotation(q,annotation)
        # 步骤3：保留原始人工观察版本；完成态初审拒绝无声覆盖。
        target=out/'annotations'/f'{qid}.json'
        if target.exists() and read(target).get('status')=='initial_reviewed' and read(target)!=annotation:
            if not revision_reason:raise ValueError('初审已存在，请另存修订版本并记录理由')
            previous=digest(target)
            write(out/'annotation_history'/qid/f'{previous}.json',read(target))
            original=out/'observations'/f'{qid}.v1.json'
            if original.exists():write(out/'annotation_history'/qid/f'{previous}.observations.json',read(original))
            write(out/'annotation_history'/qid/f'{previous}.revision.json',dict(reason=revision_reason,previous_sha256=previous))
        write(out/'observations'/f'{qid}.v1.json',spec);write(target,annotation)
