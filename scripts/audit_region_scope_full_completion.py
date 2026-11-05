"""exp27全量独立核验：只读原始返回/冻结输入与统计，不调用选择器或模型。"""
import argparse
from collections import Counter
from functools import lru_cache
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import rows,durable,utc
from videoqa_lmms_bridge.common import load_task,task_doc


@lru_cache(maxsize=None)
def file_hash(path):
    """接口：本次只读核验内对重复引用去重；不跨任务持久复用来源摘要。"""
    return sha256(path)


def interval(data,left,right):
    """接口：独立实现按时长分层、视频簇重抽，复算10000次固定随机流。"""
    rng=np.random.default_rng(2027);num=np.zeros(10000);den=np.zeros(10000)
    # 步骤1同视频差值聚合；步骤2各层成对抽簇，最后求百分位。
    for layer in ('short','medium','long'):
        videos={}
        for r in data:
            if r['stratum']==layer:videos.setdefault(r['video_id'],[]).append(int(left[r['question_id']])-int(right[r['question_id']]))
        if not videos:continue
        a=list(videos.values());s=np.array([sum(v) for v in a]);n=np.array([len(v) for v in a]);draw=rng.integers(len(a),size=(10000,len(a)))
        num+=s[draw].sum(1);den+=n[draw].sum(1)
    return np.percentile(100*num/den,[2.5,97.5]).tolist()


def verify(root):
    """接口：读物理原文独立计分、所有主比较/子集/题型/区间，并核验来源与额度。"""
    root=Path(root);summary=read_json(root/'summary.json');data=rows();ids={r['question_id'] for r in data};method='RD-1.2-exp27';total=0;profiles={};comparisons=0
    dev=read_json(ROOT/'data/manifests/video_mme_development_200_20260920.json')['rows'];dq={r['question_id'] for r in dev};dv={r['video_id'] for r in dev}
    for profile in ('qwen32','llava8'):
        run=root/'qa'/profile;spec=read_json(run/'protocol.json');fp=sha256(run/'protocol.json');current={};kinds=Counter();input_groups={};frame_counts={}
        assert set(spec['question_ids'])==ids
        for r in data:
            q=r['question_id'];v=read_json(run/'results'/method/(q+'.json'));item=read_json(run/'inputs'/(q+'.json'));ref=item['groups'][0]['reference'];obs=v['answer']['observed']
            assert v['question_id']==q and v['protocol_sha256']==fp and v['selection']==item['items']['S']['selection']
            assert v['rgb_sha256']==item['items']['S']['rgb_sha256'];actual=item['items']['S']['processed']
            assert obs.get('processor_pixel_sha256',obs.get('pixel_sha256'))==actual['pixel_sha256'] and obs['input_token_ids']==actual['input_token_ids'] and obs['prompt']==actual['prompt']
            frame_counts.setdefault(method,{})[q]=len(v['selection']['selected_frames'])
            keys=[(f['source_pts'],f['source_frame_index']) for f in v['selection']['selected_frames']];assert keys==sorted(keys) and len(set(keys))==len(keys)
            assert obs['source_frames']==len(keys) and obs['generate_calls']==obs['prefill_calls']==1
            if profile=='llava8':assert obs['visual_tokens']==210*len(keys) and obs['max_new_tokens']==16
            else:assert obs['video_grid_thw']==actual['video_grid_thw'] and obs['internal_padding_frames']==len(keys)%2 and obs['generation_kwargs']['max_new_tokens']==16 and obs['encoding_fps']==2
            if ref is None:
                raw_path=run/'raw_generations'/method/(q+'.json');call=read_json(run/'calls'/method/(q+'.json'));assert call['status']=='completed' and call['answer']==v['answer'];total+=1
            else:
                assert v['answer']==read_json(ref['path'])['answer'];paths=[p for p in ref['sources'] if '/raw_generations/' in p];assert len(paths)==1;raw_path=Path(paths[0])
            raw=read_json(raw_path);assert raw['generation_state']==dict(started=True,returned=True)
            assert raw['raw_generated_text'].strip()==v['answer']['raw_output'].strip()
            metric=load_task().videomme_process_results(task_doc(r),[raw['raw_generated_text']])['videomme_perception_score']
            correct=metric['answer']==metric['pred_answer'];assert correct==v['correct'];current[q]=correct;kinds[item['reference_kind']]+=1;input_groups[q]=item['reference_kind']
            for p,h in item['sources'].items():assert file_hash(p)==h
        assert dict(kinds)==spec['reference_counts']
        historical={}
        for method_ref in ('BASE-Uniform','BASE-BLIP-TopK','RD-1.2-exp07','RD-1.2-exp12'):
            candidates={p:h for p,h in summary['source_result_hashes'].items() if '/'+method_ref+'/' in p}
            group={}
            for p,h in candidates.items():
                if ('llava8' in p)!=(profile=='llava8'):continue
                assert file_hash(p)==h;v=read_json(p);group[v['question_id']]=v['correct'];frame_counts.setdefault(method_ref,{})[v['question_id']]=len(v['selection']['selected_frames'])
            assert set(group)==ids,(profile,method_ref,len(group));historical[method_ref]=group
        groups={**historical,method:current};layers={'all2700':data,'development200':[r for r in data if r['question_id'] in dq],
            'other2500':[r for r in data if r['question_id'] not in dq],'outside_development_videos2100':[r for r in data if r['video_id'] not in dv]}
        for kind in ('development','parent','new'):layers['input_'+kind]=[r for r in data if input_groups[r['question_id']]==kind]
        layers['common_full_budget']=[r for r in data if all(frame_counts[m][r['question_id']]==spec['frame_budget'] for m in groups)]
        for m,g in groups.items():
            metrics=summary['profiles'][profile]['metrics'][m]
            for name,rs in {**layers,**{s:[r for r in data if r['stratum']==s] for s in ('short','medium','long')}}.items():
                assert metrics[name]['n']==len(rs) and metrics[name]['correct']==sum(g[r['question_id']] for r in rs)
            for task,val in metrics['official_tasks'].items():
                rs=[r for r in data if r['task_type']==task];assert val['n']==len(rs) and val['correct']==sum(g[r['question_id']] for r in rs)
        for ref,scopes in summary['profiles'][profile]['comparisons'].items():
            for name,pair in scopes.items():
                rs=layers[name];cells={k:[] for k in ('both_correct','left_only','right_only','both_wrong')}
                for r in rs:
                    q=r['question_id'];a,b=current[q],historical[ref][q];cells['both_correct' if a and b else 'left_only' if a else 'right_only' if b else 'both_wrong'].append(q)
                assert pair['cells']==cells and pair['n']==len(rs)
                assert abs(pair['difference_pp']-100*(len(cells['left_only'])-len(cells['right_only']))/len(rs))<1e-10
                assert np.allclose(interval(rs,current,historical[ref]),pair['stratified_video_cluster_bootstrap_95'],atol=1e-10,rtol=0);comparisons+=1
        profiles[profile]=dict(results=len(current),correct=sum(current.values()),partition=dict(kinds))
    assert total==1096
    value=dict(status='passed',profiles=profiles,new_physical_calls=total,comparisons_recomputed=comparisons,read_only=True,new_model_calls=0,new_selection_calls=0,utc=utc())
    durable(root/'independent_acceptance.json',value);return value


def main():
    """接口：只接受已完成运行目录，禁止启动选择或生成。"""
    p=argparse.ArgumentParser();p.add_argument('run');a=p.parse_args();print(verify(a.run))


if __name__=='__main__':main()
