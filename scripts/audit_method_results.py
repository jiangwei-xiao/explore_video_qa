"""事后证据复核：只读取结果、特征和源码，不重新调用分类或问答模型。"""
import argparse
import argparse
import copy
import json
import os
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from videoqa_runtime.common import ROOT,read_json,write_json,sha256,offline_environment
from videoqa_runtime.protocol import question_text
from videoqa_methods.records import VARIANTS,validate_group,result_path,check_recovery
from videoqa_methods.algorithm import segment_scores,compete,hotspot_candidates,reselect
from videoqa_methods.model_ops import parse_scope,SCOPE_PROMPT


def audit(run):
    """接口：复核200条结果及100次分类、重放选择机制并写入审计报告；输入为已完成运行目录。"""
    # 步骤1：检查执行指纹、历史基线和所有结果/特征文件的身份。
    run=Path(run); spec=read_json(run/'protocol.json'); cfg=spec['config']
    for name,digest in spec['execution_code_sha256'].items():
        assert sha256(ROOT/name)==digest==sha256(run/'code_snapshot'/name),name
    baseline=ROOT/'outputs/baselines/videomme50_uniform_topk_20260914_r1'
    assert sha256(baseline/'protocol.json')==spec['baseline_protocol_sha256']
    assert sha256(baseline/'summary.json')==spec['baseline_summary_sha256']
    assert sha256(ROOT/'requirements-runtime.lock.txt')==spec['runtime_lock_sha256']
    rows=read_json(ROOT/'data/manifests/video_mme_development.json')['rows']
    assert len(check_recovery(run,rows,sha256(run/'protocol.json')))==200
    offline_environment()
    from llava.conversation import conv_templates
    shared_inputs=defaultdict(list); scope_types=defaultdict(Counter)
    coarse_max_difference=0.; gain_max_difference=0.; max_unattributed=0.
    for row in rows:
        qid=row['question_id']
        group={v:read_json(result_path(run,v,qid)) for v in VARIANTS}
        base=read_json(baseline/'results/topk'/f'{qid}.json')
        validate_group(group,row,sha256(run/'protocol.json'),base)
        assert sha256(result_path(run,'C',qid))==group['D']['donor_result_sha256']
        # 步骤2：核对纯视觉特征、PTS关系，并重放分段、竞争与补查后重选。
        for v,r in group.items():
            p=r['pool']; n=p['initial_count']; f=np.load(run/r['feature_file'],allow_pickle=False)
            assert f.dtype==np.float32 and f.shape==(len(p['candidates']),768)
            assert np.isfinite(f).all() and np.allclose(np.linalg.norm(f,axis=1),1.,atol=1e-5)
            assert all(r['answer_state'][key] for key in ('started','returned'))
            tb=Fraction(p['video']['time_base']); origin=p['video']['start_pts']
            for candidate in p['candidates']:
                assert abs(candidate['timestamp_seconds']-float((candidate['source_pts']-origin)*tb))<1e-8
                assert candidate['timestamp_seconds']+1e-8>=candidate['requested_seconds']
            if v!='D':
                difference=float(np.max(np.abs(np.asarray(p['scores'][:n])-np.asarray(base['selection']['scores']))))
                coarse_max_difference=max(coarse_max_difference,difference)
                seg,wave=segment_scores(p['scores'][:n],p['candidates'][:n],p['video'],cfg)
                assert seg==p['segments'] and wave['peaks']==p['wavelet']['peaks']
                chosen,quotas,trace=compete(p['candidates'][:n],p['scores'][:n],f[:n],seg,p['lambda_value'],cfg)
                assert chosen==p['coarse_selected'] and quotas==p['quotas']
                for saved,replayed in zip(p['competition_trace'],trace):
                    assert saved['candidate_index']==replayed['candidate_index']
                    gain_max_difference=max(gain_max_difference,abs(saved['gain']-replayed['gain']))
                if v=='C':
                    hot,windows=hotspot_candidates(p['candidates'][:n],p['scores'][:n],f[:n],chosen,seg,quotas,p['video'],cfg)
                    assert hot==p['hotspot_trace'] and len(windows)<=2
                    fine_counts=Counter()
                    for candidate in p['candidates'][n:]:
                        fine_counts[candidate['hotspot_index']]+=1
                        target=Fraction(candidate['requested_fraction'])
                        assert (target*4).denominator==1 and int(target*4)%4!=1
                        h=windows[candidate['hotspot_index']]
                        actual=(candidate['source_pts']-origin)*tb
                        assert Fraction(h['left_fraction'])<=actual<=Fraction(h['right_fraction'])
                    assert all(count<=12 for count in fine_counts.values()) and sum(fine_counts.values())<=24
                    final,_=reselect(p['candidates'],p['scores'],f,seg,quotas,chosen,n,p['lambda_value'],cfg)
                    assert final==p['selected_indices']
            # 步骤3：最终问答提示词严格与旧接口一致；分类只含原问题并独立计数。
            text=question_text(row['question'],row['options'],p['video']['duration_seconds'],[x['timestamp_seconds'] for x in p['selected_frames']])
            conv=copy.deepcopy(conv_templates['qwen_1_5']); conv.append_message(conv.roles[0],text); conv.append_message(conv.roles[1],None)
            assert conv.get_prompt()==r['answer']['prompt']
            if v in ('B','C'):
                s=p['scope']; label,fallback=parse_scope(s['raw_output'])
                assert label==s['label'] and fallback==s['fallback'] and s['visual_inputs']==0
                conv=copy.deepcopy(conv_templates['qwen_1_5']); conv.append_message(conv.roles[0],SCOPE_PROMPT.format(question=row['question'])); conv.append_message(conv.roles[1],None)
                assert conv.get_prompt()==s['prompt'] and all(r['scope_state'][key] for key in ('started','returned'))
                if v=='B': scope_types[row['task_type']][label]+=1
            else:
                assert not r['scope_state']
            key=(qid,tuple(x['source_pts'] for x in p['selected_frames']),r['answer']['prompt'])
            shared_inputs[key].append((v,r['answer']['parsed_answer']))
            max_unattributed=max(max_unattributed,r['timings']['unattributed_seconds'])
    # 步骤4：确认正常调用数；同输入预测不一致时记录，不按答题输赢重跑。
    attempts=[read_json(p) for p in (run/'attempts').glob('*/*.json')]
    counts=dict(attempts=len(attempts),qa_returned=sum(a.get('answer_state',{}).get('returned',False) for a in attempts),
                scope_returned=sum(a.get('scope_state',{}).get('returned',False) for a in attempts),
                failed_attempts=sum(a['status']!='completed' for a in attempts))
    assert counts['qa_returned']==200 and counts['scope_returned']==100
    disagreements=[dict(question_id=key[0],predictions=values) for key,values in shared_inputs.items() if len({v[1] for v in values})>1]
    result=dict(status='passed',counts=counts,coarse_score_max_abs_difference=coarse_max_difference,
                competition_gain_max_abs_difference=gain_max_difference,maximum_unattributed_seconds=max_unattributed,
                mechanism_replay='200 selections verified; A/B/C coarse and C refinement replayed',
                prompts='200 QA and 100 scope prompts verified',identical_input_prediction_disagreements=disagreements,
                scope_by_dataset_task_type_posthoc={k:dict(v) for k,v in scope_types.items()},
                note='题型仅用于事后描述，不进入分类或选择器；不一致预测仅记录不重试。')
    write_json(run/'review.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='方法实验事后证据复核，不调用模型')
    parser.add_argument('run_dir',type=Path)
    audit(parser.parse_args().run_dir)
