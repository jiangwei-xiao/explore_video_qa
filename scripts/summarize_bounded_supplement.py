"""exp19配对汇总：复算真实答案、固定对照和输入一致性，不追加任何生成。"""
import csv
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from videoqa_runtime.common import read_json, sha256
from videoqa_full.state import durable
from videoqa_full.report import compare
from videoqa_budget32.development import verify, selected_rows
from videoqa_budget32.run import validate
from videoqa_audit.query_input_probe import BASE, FULL


def main():
    """接口：全部200条经正式校验器核验；按冻结题序列完整得失，不用局部诊断替代成绩。"""
    method = 'RD-1.2-exp19'
    run = ROOT/'outputs/diagnostics/rd12_exp19__qwen32__dev200__20261003_r2'
    output = ROOT/'outputs/methods/rd12_exp19__qwen32__dev200__20261003_r2'
    spec = verify(run); data = selected_rows(spec)
    references = {'RD-1.2-exp07': FULL/'imported/results/RD-1.2-exp07',
        'RD-1.2-exp12': ROOT/'outputs/diagnostics/rd12_exp12__qwen32__dev200__20261002_r1/results/RD-1.2-exp12',
        'RD-1.2-exp17': ROOT/'outputs/diagnostics/rd12_exp17__qwen32__dev200__20261003_r1/results/RD-1.2-exp17',
        'BASE-Uniform': BASE/'results/BASE-Uniform', 'BASE-BLIP-TopK': BASE/'results/BASE-BLIP-TopK'}
    sources, results, refs = {}, {}, {name: {} for name in references}
    # 步骤1：预测、模型输入、帧预算、提示词和调用账本都由正式路径复验。
    for row in data:
        q = row['question_id']; result = validate(run, row, method, spec)
        results[q] = result
        assert result['correct'] == (result['answer']['parsed_answer'] == 'ABCD'[row['answer_index']])
        path = run/'results'/method/(q+'.json'); sources[str(path)] = sha256(path)
        for name, directory in references.items():
            p = directory/(q+'.json'); old = read_json(p)
            assert old['question'] == row['question'] and old['options'] == row['options']
            assert old['correct'] == (old['answer']['parsed_answer'] == 'ABCD'[row['answer_index']])
            refs[name][q] = old; sources[str(p)] = sha256(p)
    # 步骤2：全部配对四格与固定bootstrap，不把有歧义321-2从官方分母删除。
    paired = {name: compare(data, results, old) for name, old in refs.items()}
    identical = {}
    for name, old in refs.items():
        ids = [q for q in results if results[q]['rgb_sha256'] == old[q]['rgb_sha256']]
        differences = [q for q in ids if results[q]['answer']['parsed_answer'] != old[q]['answer']['parsed_answer']]
        identical[name] = dict(question_ids=ids, prediction_differences=differences)
    cells = paired['RD-1.2-exp07']['cells']
    # 步骤3：固定复核顺序，前两改善/退化；歧义项不作干净机制案例，原计分仍保留。
    panel = [q for q in cells['left_only'] if q not in ['321-2', '491-2']][:2] + cells['right_only'][:2]
    summary = read_json(run/'summary.json')
    correct = sum(r['correct'] for r in results.values())
    assert correct == summary['results']['all']['correct']
    decision = (f'{correct}/200 below exp07; no full test or automatic parameter sweep'
                if correct < sum(r['correct'] for r in refs['RD-1.2-exp07'].values())
                else 'Requires separate mechanism review; no automatic full test')
    result = dict(status='completed', method=method, metrics=summary['results'], comparisons=paired,
                  identical_rgb_checks=identical, new_answer_calls=200,
                  parse_failures=summary['parse_failures'], qa_seconds=summary['qa_seconds'],
                  case_review_ids=panel, data_ambiguities=['321-2', '491-2'],
                  audit=read_json(run/'completion_audit.json'),
                  decision=decision, sources=sources)
    dest = output/'paired_comparison.json'
    if dest.exists():
        assert read_json(dest) == result
    else:
        durable(dest, result)
    with (output/'paired_answers.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['question_id', 'stratum', 'method', 'prediction', 'correct'])
        writer.writeheader()
        for row in data:
            q = row['question_id']
            for name, values in {method: results, **refs}.items():
                writer.writerow(dict(question_id=q, stratum=row['stratum'], method=name,
                                     prediction=values[q]['answer']['parsed_answer'], correct=values[q]['correct']))
    print(summary['results'], flush=True)
    print({name: dict(gained=len(p['cells']['left_only']), lost=len(p['cells']['right_only']),
                     difference_pp=p['difference_pp'], interval=p['stratified_video_cluster_bootstrap_95'])
           for name, p in paired.items()}, flush=True)
    print('Review panel', panel, 'same RGB checks', identical, flush=True)


if __name__ == '__main__':
    main()
