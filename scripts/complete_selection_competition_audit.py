"""补齐双数据集复盘的全量实际竞争轨迹；独立派生目录，不改上一阶段快照。"""
import os
os.environ.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable,exclusive,utc
from videoqa_audit.selection_competition_audit import audit_competition
from analyze_cross_dataset_review import Reader,pool_pair,FACTS,verify


def save_compressed(path,value):
    """接口：完整逐步轨迹压缩原子保存，mtime固定以支持恢复一致性；不写旧选择文件。"""
    raw=gzip.compress(json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode(),mtime=0)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_bytes()!=raw:raise ValueError('Completed competition output changed')
    else:
        temporary=path.with_suffix('.tmp')
        temporary.write_bytes(raw);os.replace(temporary,path)
    return hashlib.sha256(raw).hexdigest()


def execute(parent):
    """接口：9748条全部补齐，事实75项单列；恢复核查来源/新产物，不触发选帧或模型。"""
    parent=Path(parent);out=parent/'completion_audit';out.mkdir(parents=True,exist_ok=True)
    with exclusive(out/'run.lock'):
        if (out/'summary.json').exists():
            old=read_json(out/'summary.json')
            for p,h in old['sources'].items():
                if sha256(p)!=h:raise ValueError('Completed source changed: '+p)
            for p,h in old['exports'].items():
                if sha256(out/p)!=h:raise ValueError('Competition export changed: '+p)
            return dict(status='passed',completed_resume_new_computations=0,new_model_calls=0,
                        rows=old['rows'],targets=old['targets'],max_component_error=old['max_component_error'])
        # 步骤1：先核验原阶段闭合与原期望特征摘要，绑定新代码，不改变原协议或结论。
        verify(parent)
        old_sources=read_json(parent/'sources.json')
        frozen=read_json(parent/'frozen_asset_acceptance.json')['files']
        files=read_json(parent/'summary.json')['exports']
        facts=read_json(FACTS)['entries'];panel=read_json(parent/'panel.json')['questions']
        if len(facts)!=75:raise ValueError('Frozen fact count changed')
        panel_ids={(x['dataset'],p,x['question_id']) for x in panel for p in x['profiles']}
        reader=Reader();sources={str(parent/'summary.json'):sha256(parent/'summary.json'),
            str(parent/'sources.json'):sha256(parent/'sources.json'),str(FACTS):sha256(FACTS),
            str(parent/'frozen_asset_acceptance.json'):sha256(parent/'frozen_asset_acceptance.json'),
            str(Path(__file__)):sha256(Path(__file__)),
            str(ROOT/'src/videoqa_audit/selection_competition_audit.py'):sha256(ROOT/'src/videoqa_audit/selection_competition_audit.py')}
        exports={};profiles={};seen_features={};count=target_count=0;max_error=0.;began=time.perf_counter()
        # 步骤2：来源从已冻结逐题清单读取，映射原K32特征至K8池，绝不补算视觉或ITM。
        for dataset in ('videomme','mlvu'):
            for profile in ('llava8','qwen32'):
                totals=Counter();native_errors=[]
                for pp in sorted((parent/'records'/dataset/profile).glob('*.json')):
                    if sha256(pp)!=files[str(pp.relative_to(parent))]:raise ValueError('Parent record differs')
                    record=read_json(pp);q=record['question_id']
                    pool,a,b,pool_path,feature_path=pool_pair(dataset,profile,q,reader)
                    if str(feature_path) not in seen_features:
                        expected=frozen.get(str(feature_path))
                        if expected is None or sha256(feature_path)!=expected:raise ValueError('Original feature fingerprint differs')
                        seen_features[str(feature_path)]=expected
                    raw=np.load(feature_path,mmap_mode='r',allow_pickle=False)
                    feature=raw[pool['original_candidate_indices']] if dataset=='videomme' and profile=='llava8' else raw
                    targets={p['candidate_index'] for p in record['structure']['topk_paths']
                             if p['exp07']['state']!='selected' or p['exp12']['state']!='selected'}
                    pts_to_id={c['source_pts']:i for i,c in enumerate(pool['candidates'][:pool['initial_count']])}
                    fact_map=[]
                    if dataset=='videomme':
                        for fact in [f for f in facts if f['question_id']==q]:
                            indices=[pts_to_id[x] for x in fact['witness_pts'] if x in pts_to_id]
                            targets.update(indices)
                            fact_map.append(dict(fact_id=fact['fact_id'],witness_indices=indices,
                                outside_pool_pts=[x for x in fact['witness_pts'] if x not in pts_to_id]))
                    result=audit_competition(pool,feature,a,b,targets)
                    # 把未入Top-K的已知见证也映射至完整粗排名，不从答案决定处理对象。
                    ranking=sorted(range(pool['initial_count']),key=lambda i:(-pool['initial_scores'][i],
                        pool['candidates'][i]['source_pts'],pool['candidates'][i]['candidate_index']))
                    ranks={i:k+1 for k,i in enumerate(ranking)}
                    for entry in result['records']:entry['global_coarse_rank']=ranks[entry['candidate_index']]
                    result.update(question_id=q,dataset=dataset,profile=profile,facts=fact_map,
                        parent_record_sha256=sha256(pp),feature_sha256=seen_features[str(feature_path)])
                    dest=out/'trajectories'/dataset/profile/(q+'.json.gz')
                    exports[str(dest.relative_to(out))]=save_compressed(dest,result)
                    if (dataset,profile,q) in panel_ids:
                        dest=out/'panel'/dataset/profile/(q+'.json');durable(dest,result)
                        exports[str(dest.relative_to(out))]=sha256(dest)
                    count+=1;target_count+=len(targets)
                    max_error=max(max_error,result['max_component_error'])
                    totals['questions']+=1;totals['target_coarse_frames']+=len(targets)
                    totals['fact_entries']+=len(fact_map)
                    totals['competition_steps']+=sum(len(x['trajectory']) for x in result['records'])
                    for entry in result['records']:
                        for name,p in entry['parents'].items():totals[name+'/'+p['terminal']['mode']]+=1
                    if count%100==0:durable(out/'status.json',dict(stage='competition',completed=count,total=9748,new_model_calls=0,utc=utc()))
                    del feature,raw
                profiles[dataset+'/'+profile]=dict(totals)
        # 步骤3：复核读取JSON与历史期望摘要相符；输出完整分母和数值误差，不声称新成绩。
        if count!=9748 or any(v['questions']!=(2700 if k.startswith('videomme') else 2174) for k,v in profiles.items()):
            raise ValueError('Complete competition denominator differs')
        for p,h in reader.sources.items():
            if p not in old_sources or h!=old_sources[p]:raise ValueError('Historical trace differs from previous acceptance')
        sources.update(reader.sources);sources.update(seen_features)
        summary=dict(status='passed',rows=count,targets=target_count,profiles=profiles,
            max_component_error=max_error,tolerance=1e-6,feature_files=len(seen_features),
            sources=sources,exports=exports,new_model_calls=0,new_answers=0,
            selection_algorithm_executed=False,seconds=time.perf_counter()-began,utc=utc())
        durable(out/'summary.json',summary)
        durable(out/'status.json',dict(status='completed',completed=count,new_model_calls=0,utc=utc()))
        return {k:v for k,v in summary.items() if k not in ('sources','exports')}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True);args=parser.parse_args()
    if Path(args.run_id).name!=args.run_id or args.run_id in ('.','..'):raise ValueError('Invalid run ID')
    print(json.dumps(execute(ROOT/'outputs/analysis'/args.run_id),ensure_ascii=False))
