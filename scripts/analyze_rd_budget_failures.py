"""冻结16/32帧结果的错误定位；只读历史资产，不调用任何模型。"""
import argparse
import csv
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from videoqa_runtime.common import read_json, sha256
from videoqa_full.state import durable, utc

SOURCE = ROOT / 'outputs/full_videoqa/uniform_topk_rd12__qwen25vl__lmms_ref_v1__videomme2700__k32_20260924__r01'
DEVELOPMENT = ROOT / 'data/manifests/video_mme_development_200_20260920.json'
METHODS = ('BASE-Uniform', 'BASE-BLIP-TopK', 'RD-1.2')


def identity(frames):
    """接口：同视频内按真实源编号和PTS对应，不混用候选编号。"""
    return {(f['source_frame_index'], f['source_pts']) for f in frames}


def average(records, key):
    """接口：空组保留空值，避免把没有样本解释成零成本或零错误。"""
    return sum(r[key] for r in records) / len(records) if records else None


def analyze(output):
    """接口：全量配对、开发子集和结构代理分组，人工证据不参与选帧。"""
    # 步骤1：绑定冻结输出与清单；恢复仅接受相同来源，不改历史实验。
    paths = [SOURCE/'per_question.csv', SOURCE/'stage_analysis/per_question.json', DEVELOPMENT]
    sources = {str(p): sha256(p) for p in paths}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/'summary.json').exists():
        prior = read_json(output/'summary.json')
        assert prior['sources'] == sources and prior['script_sha256'] == sha256(__file__)
        assert prior['records_sha256'] == sha256(output/'per_question.json')
        return prior
    rows = list(csv.DictReader((SOURCE/'per_question.csv').open()))
    groups = {}
    for r in rows:
        g = groups.setdefault(r['configuration'], {})
        assert r['question_id'] not in g
        g[r['question_id']] = dict(correct=r['correct']=='True', prediction=r['parsed_answer'])
    assert len(groups) == 6 and all(len(g)==2700 for g in groups.values())
    dev = {r['question_id'] for r in read_json(DEVELOPMENT)['rows']}
    assert len(dev) == 200
    stages = read_json(SOURCE/'stage_analysis/per_question.json')
    records = []
    # 步骤2：归入八种三方法对错模式；仅描述结构，不以答错自动标注证据缺失。
    for stage in stages:
        q = stage['question_id']; selections = {}; exports = {}
        for m in METHODS:
            p = SOURCE/'selections'/m/(q+'.json')
            exports[m] = sha256(p)
            selections[m] = identity(read_json(p)['selection']['selected_frames'])
        old, new = stage['old'], stage['new']
        prediction = {k:v[q] for k,v in groups.items()}
        pattern = ''.join('1' if prediction['K32__'+m]['correct'] else '0' for m in METHODS)
        records.append(dict(question_id=q,video_id=stage['video_id'],stratum=stage['stratum'],
            task_type=stage['task_type'],development=q in dev,pattern_UTR=pattern,predictions=prediction,
            region_count=new['region_count'],window_seconds=new['window_seconds'],
            window_growth=new['window_seconds']-old['window_seconds'],
            selected_fine=new['selected_new_count'],fine_candidates=new['new_candidate_count'],
            seeds_retained=new['seeds_retained'],topk_overlap=len(selections[METHODS[1]]&selections[METHODS[2]]),
            uniform_overlap=len(selections[METHODS[0]]&selections[METHODS[2]]),
            old_final_retained=stage['old_retained_in_full32'],
            fixed_old_pool_retained=stage['old_retained_in_fixed_pool32'],selection_sha256=exports))
    assert len(records)==2700 and len({r['question_id'] for r in records})==2700
    # 步骤3：开发与全量分开报告，逐组分母及全部题号保留，不能把关联当因果。
    reports={}
    for name, subset in [('all2700',records),('development200',[r for r in records if r['development']])]:
        report=dict(n=len(subset),accuracy={k:sum(r['predictions'][k]['correct'] for r in subset) for k in groups},patterns={})
        for pattern in sorted({r['pattern_UTR'] for r in subset}):
            xs=[r for r in subset if r['pattern_UTR']==pattern]
            report['patterns'][pattern]=dict(n=len(xs),question_ids=[r['question_id'] for r in xs],
                duration=dict(Counter(r['stratum'] for r in xs)),task=dict(Counter(r['task_type'] for r in xs)),
                mean={k:average(xs,k) for k in ('region_count','window_seconds','window_growth','selected_fine','fine_candidates','seeds_retained','old_final_retained')})
        report['pairs']={}
        pairs=[('K32__RD-1.2','K32__'+m) for m in METHODS[:2]]+[('K32__RD-1.2','K16__RD-1.2')]
        for a,b in pairs:
            gain=[r['question_id'] for r in subset if r['predictions'][a]['correct'] and not r['predictions'][b]['correct']]
            loss=[r['question_id'] for r in subset if not r['predictions'][a]['correct'] and r['predictions'][b]['correct']]
            report['pairs'][a+'_minus_'+b]=dict(gain=gain,loss=loss,net=len(gain)-len(loss))
        reports[name]=report
    durable(output/'per_question.json',records)
    summary=dict(status='completed',utc=utc(),new_model_calls=0,sources=sources,script_sha256=sha256(__file__),
        records_sha256=sha256(output/'per_question.json'),reports=reports,
        limitations=['UTR pattern = Uniform/TopK/RD correctness; not evidence labels',
                      'Existing full set has been inspected; not an unseen test',
                      'Group structural means are associations, not causal attribution'])
    durable(output/'summary.json',summary)
    return summary


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    result=analyze(args.output)
    for name, report in result['reports'].items():
        print(name,report['accuracy'],flush=True)
        print('patterns', {k:v['n'] for k,v in report['patterns'].items()},flush=True)
        print('pairs',{k:{'gain':len(v['gain']),'loss':len(v['loss']),'net':v['net']} for k,v in report['pairs'].items()},flush=True)
