"""保存本轮AI已完成的查询语义和实际变化帧review，不修改冻结代理或实验记录。"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from videoqa_runtime.common import read_json, write_json, sha256
from videoqa_methods.followups import utc

CASE_NOTES={
 '004-3':dict(judgment='支持窗口外保护的针对性作用，但组合换帧不能单独证明84秒帧因果',
             observation='E恢复84.28秒Neanderthalensi完整标签；C移除的94.52秒是Modern阶段。E还改变31.28和70/70.76秒等画面。答案B错误→D正确；82.28秒依旧不能声称有阶段文字。',
             evidence_times=[31.28,70.,70.76,84.28,94.52]),
 '260-2':dict(judgment='答题恢复但关键证据改善未获证明',
             observation='C/E差异仅在8.258/9.009/9.259/9.760秒衣架及面具画面。61.27秒后续动作帧没有在此次换入，不将D错误→B正确解释为补回喷雾；提示整组上下文敏感性。',
             evidence_times=[8.258,9.009,9.259,9.760]),
 '103-1':dict(judgment='共同失败未解决，不能将不同错误选项当成机制收益',
             observation='E以26.276秒进食画面替换C的27.527秒同场景；叉子关键候选5仍未进入。E与C均答C，B答B，均错。',
             evidence_times=[5.,26.276,27.527]),
 '893-1':dict(judgment='恢复已列部分事实，无新增正确题',
             observation='E保留699.28秒冷冻/冷藏/新鲜蔬菜对比图，也换入1099.28秒饮食宣传电视画面；C/E均正确，不宣称比较图带来准确率提升。',
             evidence_times=[699.28,1099.28]),
 '870-1':dict(judgment='恢复活动执行画面仍答错，范围保护不足以解决未出现判断',
             observation='E保留901.275秒迷你高尔夫设施和参与者；移除862.028/863.529秒挑战标题相近画面，另保留840.256秒道路。仍答错，不把该单个活动见证认定为所有挑战关系充分。',
             evidence_times=[840.256,862.028,863.529,901.275]),
 '871-2':dict(judgment='冻结目录的见证损失是代理假阳性，不确认真实事实损失',
             observation='自动表因921.000秒不在E中而标记green_toy丢失；E的919.767秒仍清楚可见猫围绿色骨状玩具，与C的921秒是相同对象/活动。查看E完整16帧后确认该信息仍在。不修改r6目录和自动CSV，另存本复核覆盖说明。',
             evidence_times=[584.267,585.267,919.767,921.])}


def record(run):
    """接口：绑定已查看的50查询与6题图板哈希；保留自动统计和AI语义判断的区别。"""
    run=Path(run); output=run/'review'; output.mkdir(exist_ok=True)
    questions=[]
    for p in sorted((run/'queries').glob('*.json')):
        q=read_json(p); qid=q['question_id']
        if q['fallback']:
            judgment='actual_query_is_original_question'
            note='已查看原始输出：非白名单内容词与原题不一致，回退正确生效。实际查询为原题，不算有效改写。'
        elif qid in ('305-3','729-1'):
            judgment='faithful_but_unchanged'
            note='输出原问题，语义忠实但没有提供表达干预，BLIP编码不变。'
        elif qid=='491-2':
            judgment='semantic_drift_unsupported_order_assertion'
            note='词汇集合通过不等于语义忠实：把询问顺序转为给定(a)(b)(c)顺序的陈述。原题还有重复事件歧义，不能作为合格有效改写。保留冻结实际查询与评分，不事后润色/回退。'
        else: raise ValueError('Unexpected lexical pass requires review')
        questions.append(dict(question_id=qid,query_sha256=sha256(p),judgment=judgment,note=note,
                              fallback=q['fallback'],reviewer='AI initial review; no user agreement claimed'))
    if len(questions)!=50: raise ValueError('Review must cover all 50')
    write_json(output/'query_fidelity.json',dict(status='completed_with_drift',questions=questions,
        raw_outputs_read=50,actual_fallbacks=47,unchanged_valid=2,changed_with_semantic_drift=['491-2'],
        effective_faithful_changed_queries=0,decision='reject_current_query_generator; no formal query QA'))
    cases=[]
    for qid,note in CASE_NOTES.items():
        path=run/'cases'/qid/'changes.png'
        cases.append(dict(question_id=qid,**note,board_sha256=sha256(path),board=str(path.relative_to(run)),
                          full_E_board_reviewed=qid=='871-2',reviewer='AI initial review'))
    write_json(output/'case_review.json',dict(cases=cases,old_audit_unchanged=True,proxy_denominator_unchanged=True,
        automatic_catalog_loss_count=1,confirmed_semantic_loss_among_flagged_cases=0,
        warning='这里只复核新增变化及预定事实，不宣称已重新审核全部新选帧或所有证据充分性。'))
    write_json(output/'decisions.json',dict(E='retain_as_refinement_safety_variant; keep B as working reference',
        E_evidence='30/50 vs C28/50; same correct question set as B30/50; 004 mechanism supported, 260 unresolved',
        Q='reject_current_restricted_query_implementation; no extra QA',
        Q_evidence='47 fallback, 2 unchanged, 1 semantically drifting ambiguous query; no valid changed query on proxy population',
        next_validation='generator instruction/fidelity validation first, then frozen offline ranking; full50 QA only if predeclared gate passes and separately authorized',
        no_automatic_optimization_or_tuning=True,review_finished_at_utc=utc(),
        review_time_upper_bound_start=read_json(run/'query_freeze.json')['utc'],
        reviewer='AI; user agreement not measured'))


if __name__=='__main__': record(sys.argv[1])
