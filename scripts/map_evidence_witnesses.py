"""核对人工事实索引并生成逐方法见证表，全程不运行模型。"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from videoqa_runtime.common import read_json, write_json, sha256
from videoqa_audit.evidence_witnesses import map_question


def run(audit, spec_path):
    """输入审计目录与不可变人工事实版本，输出该版本的逐题JSON及可点击MD。"""
    audit = Path(audit); spec_path = Path(spec_path); spec = read_json(spec_path)
    # 步骤1：必须匹配联合图板清单哈希，防止题序/帧序变更造成错误归因。
    if sha256(audit / 'questions.json') != spec['questions_sha256']:
        raise ValueError('图板清单哈希不一致')
    questions = {q['question_id']: q for q in read_json(audit / 'questions.json')}
    outcomes = read_json(audit / 'outcomes.json')
    out = audit / 'full_review' / spec['version']; out.mkdir(exist_ok=True)
    frozen = out / 'spec.json'
    if frozen.exists() and read_json(frozen) != spec:
        raise ValueError('同版本事实规格不同，请新建版本而非覆盖旧判断')
    write_json(frozen, spec)
    results = []
    # 步骤2：已列出的见证必须确实经过384页查看；未查看不偷换成不确定。
    for qid, item in spec['questions'].items():
        visibility = read_json(audit / 'full_review' / 'visibility' / f'{qid}.json')
        for page in visibility['pages'].values():
            if sha256(audit / page['board_file']) != page['sha256']:
                raise ValueError('384图板哈希改变')
        reviewed = {p for page in visibility['pages'].values() for p in page['source_pts']}
        for fact in item['facts']:
            for i in fact['union_indices']:
                if i < 1 or i > len(questions[qid]['frames']) or questions[qid]['frames'][i-1]['source_pts'] not in reviewed:
                    raise ValueError('事实所指帧未完成384查看')
        result = map_question(questions[qid], outcomes[qid], item)
        write_json(out / f'{qid}.json', result); results.append(result)
    # 步骤3：报告见证计数而非充分率，链接精确输入图让读者复核。
    lines = ['# 可见事实与各组输入对应', '', '这是人工指定事实的见证表，不是充分性自动打分；空白见证不证明原视频没有该事实。', '']
    for r in results:
        lines += [f"## {r['question_id']}", '', r['question'], '', r['reasoning_limits'], '', '| 可见事实 | 均匀 | Top-K | A | B | C | D |', '|---|---:|---:|---:|---:|---:|---:|']
        methods = {g['method']: g for g in r['groups'].values()}
        for f in r['facts']:
            counts = [str(methods[m]['facts'][f['id']]['count']) for m in ['uniform','topk','A','B','C','D']]
            lines.append('| ' + f['description'] + ' | ' + ' | '.join(counts) + ' |')
        lines.append('')
        for f in r['facts']:
            links = [f"[{x['timestamp_seconds']:.3f}秒](../../frames/{r['question_id']}/{x['source_pts']}.input.png)" for x in f['witnesses']]
            lines += [f"- {f['id']}：" + ('、'.join(links) if links else '尚未列出可确认见证') + '。' + f['limitation'], '']
    (out / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    write_json(out / 'summary.json', {'question_count':len(results),'group_count':sum(len(r['groups']) for r in results),'model_calls':0,'spec_sha256':sha256(spec_path)})
    print(len(results), 'questions mapped without sufficiency inference')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--audit',required=True); p.add_argument('--spec',required=True)
    args=p.parse_args(); run(args.audit,args.spec)
