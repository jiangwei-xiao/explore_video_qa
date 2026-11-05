"""共同访问协议下的精度、逐项真实成本与条件投入门；不宣称独立远端E2E。"""
from collections import Counter
from pathlib import Path
import itertools
import math

import numpy as np
from PIL import Image
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import digest
from .trial import METHODS,freeze
from .transport import PROFILES,rgb_hash


def paired_bootstrap(left,right,groups=None,seed=2027):
    """接口：本试点每视频仅一题，视频簇重抽10000次；不称独立盲测。"""
    a=np.asarray(left,dtype=float)-np.asarray(right,dtype=float);rng=np.random.default_rng(seed)
    if groups is None:groups=['all']*len(a)
    blocks=[]
    for group in dict.fromkeys(groups):
        ids=np.flatnonzero(np.asarray(groups)==group)
        blocks.append(a[rng.choice(ids,(10000,len(ids)),replace=True)])
    values=np.concatenate(blocks,axis=1).mean(1)*100
    return dict(delta_pp=float(a.mean()*100),ci95_pp=np.quantile(values,[.025,.975]).tolist())


def budget_gate(tables):
    """接口：只按事前门槛筛选下一笔资源；所有固定链路和预算点均保留。"""
    rd='RD-1.2-exp12';decisions=[]
    for profile in ('fast','limited'):
        for budget in ('8','32'):
            values=tables[profile][budget];r=values[rd];others=[(m,values[m]) for m in METHODS if m!=rd]
            best=max(v['successful_correct'] for _,v in others)
            strongest=[(m,v) for m,v in others if v['successful_correct']==best]
            # 步骤1准确率的净增门同时约束原片字节和组件时延，不用纯网络sleep制造优势。
            gain=next((m for m,v in strongest if r['successful_correct']>=best+2 and
                r['mean_raw_bytes']<=v['mean_raw_bytes']*1.1 and
                r['mean_component_seconds']<=v['mean_component_seconds']*1.1),None)
            # 步骤2成本门须对精度不低于RD的最廉价参照成立，不能只挑最慢基线。
            candidates=[(m,v) for m,v in others if v['successful_correct']>=r['successful_correct']]
            saved=None
            if r['successful_correct']>=best and candidates:
                cheap_time=min(candidates,key=lambda x:(x[1]['mean_component_seconds'],METHODS.index(x[0])))
                cheap_bytes=min(candidates,key=lambda x:(x[1]['mean_raw_bytes'],METHODS.index(x[0])))
                for dimension,reference in [('time',cheap_time),('bytes',cheap_bytes)]:
                    m,v=reference
                    if dimension=='time':ok=r['mean_component_seconds']<=v['mean_component_seconds']*.8 and r['mean_raw_bytes']<=v['mean_raw_bytes']
                    else:ok=v['mean_raw_bytes']>0 and r['mean_raw_bytes']<=v['mean_raw_bytes']*.8 and r['mean_component_seconds']<=v['mean_component_seconds']
                    if ok:saved=dict(dimension=dimension,reference=m);break
            passed=bool(gain or saved) and r['successful_correct']>0
            decisions.append(dict(profile=profile,budget_mib=int(budget),passed=passed,accuracy_reference=gain,cost_reference=saved,
                rd_successful_correct=r['successful_correct'],best_baseline_successful_correct=best,
                statement='resource screening only; component timing and observed sample, not statistical superiority'))
    return dict(passed=any(v['passed'] for v in decisions),decisions=decisions)


def summarize(root,dataset):
    """接口：精度、原片获取、共享检索和QA分别复算；预算不足题计未完成。"""
    from .runtime import data_rows
    root=Path(root);data=data_rows(root,dataset);labels=read_json(root/'labels.json')
    replay=read_json(root/'network_replay'/dataset/'summary.json')['records'];records=[];correct={m:[] for m in METHODS};native={m:[] for m in METHODS}
    for row in data:
        q=row['question_id'];prepared=read_json(root/'prepared'/dataset/(q+'.json'));index=read_json(root/'indices'/(row['video_id']+'.json'))
        label='ABCD'[labels[dataset+'/'+q]]
        for method in METHODS:
            result=read_json(root/'qa'/dataset/'results'/method/(q+'.json'));item=prepared['items'][method]
            if result['correct']!=(result['prediction']==label):raise ValueError('Scoring mismatch')
            correct[method].append(int(result['correct']));native[method].append(int(result['native_correct']))
            requests=item['requests'];raw_bytes=sum(v['payload_bytes'] for v in requests)
            common=prepared['preview_scoring'];fine=item['extra_visual'];o=result['answer']['observed']
            common_seconds=sum(common[k] for k in ('preprocess_seconds','forward_seconds')) if method!='BASE-Uniform' else 0.
            fine_seconds=sum(fine[k] for k in ('preprocess_seconds','forward_seconds'))
            pp=root/'qa'/dataset/'preprocess_times'/result['representative']/(q+'.json')
            if not pp.exists():raise ValueError('Actual QA preprocessing cost is missing; no repeated generation')
            qa_pre=read_json(pp)['charged_actual_call_seconds']
            if qa_pre is None:raise ValueError('QA preprocessing unobserved')
            # 步骤1只合成不重叠的已测组件；不使用带审计PNG写盘的preparation_seconds。
            base=common_seconds+fine_seconds+item['selection_compute_seconds']+item['fine_preview_seconds']+o['generation_seconds']+qa_pre
            access=sum(v['decode_seconds']+v['png_encode_seconds']+v['client_unpack_seconds'] for v in requests)
            links={profile:sum(replay[profile+'/'+r['payload_sha256']]['seconds'] for r in requests) for profile in PROFILES}
            cold={profile:(0. if method=='BASE-Uniform' else replay[profile+'/'+index['preview_payload_sha256']]['seconds']) for profile in PROFILES}
            value=dict(dataset=dataset,question_id=q,method=method,task=row.get('task_type'),stratum=row.get('stratum'),
                correct=result['correct'],native_correct=result['native_correct'],prediction=result['prediction'],
                raw_payload_bytes=raw_bytes,requests=len(requests),requested_unique_original_frames=sum(len(r['requested_sources']) for r in requests),
                decoded_frames=sum(r['decoded_frames'] for r in requests),logical_original_read_bytes=sum(r['file_read_bytes'] for r in requests),
                common_blip_seconds=common_seconds,extra_blip_seconds=fine_seconds,
                coarse_itm_frames=common['frames'] if method!='BASE-Uniform' else 0,extra_itm_frames=fine['itm_frames'],
                extra_visual_frames=fine['frames'] if method=='RD-1.2-exp12' else 0,
                selection_seconds=item['selection_compute_seconds'],qa_generation_seconds=o['generation_seconds'],qa_preprocess_seconds=qa_pre,
                raw_access_compute_seconds=access,network_replay_seconds=links,
                warm_component_seconds={p:base+access+links[p] for p in PROFILES},
                first_query_preview_transfer_seconds=cold,preview_build_seconds=index['build_seconds'],
                preview_storage_bytes=sum((root/f).stat().st_size for f in index['preview_files']),
                peak_allocated_gib=o.get('peak_allocated_gib'),physical_call=result['call_id'])
            records.append(value)
    metrics={}
    for method in METHODS:
        group=[v for v in records if v['method']==method];a=np.asarray(correct[method]);p=a.mean();n=len(a);z=1.959963984540054
        den=1+z*z/n;center=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
        metrics[method]=dict(n=n,correct=int(a.sum()),accuracy=float(100*p),native_correct=sum(native[method]),native_accuracy=100*sum(native[method])/n,
            wilson95_percent=[100*(center-half),100*(center+half)],mean_raw_bytes=float(np.mean([v['raw_payload_bytes'] for v in group])),
            means={k:float(np.mean([v[k] for v in group])) for k in ('requests','requested_unique_original_frames','decoded_frames','logical_original_read_bytes',
                'common_blip_seconds','extra_blip_seconds','selection_seconds','qa_generation_seconds','qa_preprocess_seconds')},
            components={profile:dict(mean=float(np.mean([v['warm_component_seconds'][profile] for v in group])),
                p50=float(np.quantile([v['warm_component_seconds'][profile] for v in group],.5)),p90=float(np.quantile([v['warm_component_seconds'][profile] for v in group],.9))) for profile in PROFILES})
    tables={}
    for profile in PROFILES:
        tables[profile]={}
        for budget in (8,32):
            tables[profile][str(budget)]={}
            for method in METHODS:
                group=[v for v in records if v['method']==method];feasible=[v['raw_payload_bytes']<=budget*1024**2 for v in group]
                tables[profile][str(budget)][method]=dict(n=len(group),completed=sum(feasible),
                    successful_correct=sum(v['correct'] and ok for v,ok in zip(group,feasible)),
                    mean_raw_bytes=float(np.mean([v['raw_payload_bytes'] for v in group])),
                    mean_component_seconds=float(np.mean([v['warm_component_seconds'][profile] for v in group])),
                    meaning='posthoc whole-plan feasibility; not a new budget-adaptive stopping policy')
    pairs={}
    for left,right in itertools.combinations(METHODS,2):
        gains=[r['question_id'] for i,r in enumerate(data) if correct[left][i] and not correct[right][i]]
        losses=[r['question_id'] for i,r in enumerate(data) if not correct[left][i] and correct[right][i]]
        strata=[r['stratum'] if dataset=='videomme' else r['task_type'] for r in data]
        pairs[left+'__minus__'+right]=dict(paired_bootstrap(correct[left],correct[right],strata),gains=gains,losses=losses,
            both_correct=[r['question_id'] for i,r in enumerate(data) if correct[left][i] and correct[right][i]],
            both_wrong=[r['question_id'] for i,r in enumerate(data) if not correct[left][i] and not correct[right][i]])
    gate=budget_gate(tables) if dataset=='videomme' else dict(passed=None,meaning='replication only; no further automatic experiment')
    value=dict(dataset=dataset,questions=len(data),metrics=metrics,pairs=pairs,budget_tables=tables,gate=gate,records=records,
        performance_boundary='Shared-asset measured components + actual response replay; excludes index build/model load; not direct remote or independent raw-video E2E',
        mlvu_metric='equal10/task, accuracy equals 7-task M-Avg on this panel' if dataset=='mlvu' else None)
    freeze(root/'reports'/(dataset+'.json'),value)
    lines=['# '+dataset+' 受控原片访问试点结果','',
        '固定60/70题的已观察可行性面板。预览与原片获取为新协议；不覆盖历史全量成绩。',
        '组件时延由实际测量分项与真实响应网络重放组成，非独立远端E2E；模型加载、预览创建/首次传输另列。','',
        '| 方法 | 正确数 | 准确率 | 原片MiB/题 | local组件秒 | fast组件秒 | limited组件秒 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for m in METHODS:
        v=metrics[m];lines.append('| '+m+' | '+str(v['correct'])+'/'+str(v['n'])+' | '+f"{v['accuracy']:.2f}% | {v['mean_raw_bytes']/1024**2:.3f} | "+' | '.join(f"{v['components'][p]['mean']:.3f}" for p in PROFILES)+' |')
    lines+=['','进入下一数据集的预设资源门：'+str(gate.get('passed')),
        '预算表按整套访问计划是否满足8/32 MiB判定，超预算不附用完整答案，不代表实现了主动停采。',
        '全部配对得失、视频bootstrap、计分版本、预览建设与逐题成本见同名JSON。']
    path=root/'reports'/(dataset+'.md');path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and path.read_text()!='\n'.join(lines)+'\n':raise ValueError('Existing report differs')
    if not path.exists():path.write_text('\n'.join(lines)+'\n')
    return dict(dataset=dataset,gate=gate,correct={m:v['correct'] for m,v in metrics.items()})


def completion_audit(root):
    """接口：从原始Token解码文本、标签、输入像素和调用复算闭合；恢复只读。"""
    from .runtime import verify,data_rows
    from videoqa_audit.frame_group_diagnostic import restore_answer,canonical_option,observed_input
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained('/home/models/lmms-lab/LLaVA-Video-7B-Qwen2',local_files_only=True)
    root=Path(root);verify(root);labels=read_json(root/'labels.json');datasets=[];calls=0;method_results=0;synthetic=0;exports={}
    for dataset in ('videomme','mlvu'):
        rp=root/'reports'/(dataset+'.json')
        if not rp.exists():
            if dataset=='mlvu' and (root/'mlvu_not_started.json').exists():continue
            raise ValueError('Required dataset report missing')
        report=read_json(rp);datasets.append(dataset);plan=read_json(root/'qa'/dataset/'plan.json')
        summarize(root,dataset)
        if len(plan['calls'])>len(data_rows(root,dataset))*6:raise ValueError('Call budget exceeded')
        for rel,h in plan['exports'].items():
            if sha256(root/rel)!=h:raise ValueError('CPU input changed')
        seen=set();counts=Counter()
        for row in data_rows(root,dataset):
            q=row['question_id'];prepared=read_json(root/'prepared'/dataset/(q+'.json'));inp=read_json(root/'qa'/dataset/'inputs'/(q+'.json'))
            if sha256(root/'prepared'/dataset/(q+'.json'))!=inp['prepared_sha256']:raise ValueError('Prepared receipt changed')
            for rel,h in prepared['exports'].items():
                if sha256(root/rel)!=h:raise ValueError('Prepared media/trace changed')
            for method in METHODS:
                path=root/'qa'/dataset/'results'/method/(q+'.json');v=read_json(path);rawpath=root/v['raw_generation_file'];raw=read_json(rawpath)
                rep=v['representative'];cp=root/'qa'/dataset/'calls'/rep/(q+'.json');call=read_json(cp)
                if (v['question_id'],v['dataset'],v['method'])!=(q,dataset,method) or sha256(rawpath)!=v['raw_generation_sha256']:raise ValueError('Result slot or raw differs')
                if call['status']!='completed' or call['answer']!=v['answer'] or call['protocol_sha256']!=sha256(root/'protocol.json'):raise ValueError('Physical call differs')
                case=dict(dataset=dataset,profile='llava8');restored=restore_answer(case,raw,dict(row,answer_index=0))
                tokens=raw['observed']['generated_token_ids'];text=tokenizer.decode(tokens,skip_special_tokens=True).strip()
                if text!=raw['raw_generated_text'].strip():raise ValueError('Raw token decoding differs')
                pred=canonical_option(case,restored);label='ABCD'[labels[dataset+'/'+q]]
                if pred!=v['prediction'] or v['correct']!=(pred==label):raise ValueError('Independent parse/scoring differs')
                if observed_input('llava8',restored)!=inp['items'][method]['processed'] or v['input_identity']!=inp['items'][method]['identity']:raise ValueError('Actual/CPU input differs')
                for f in inp['items'][method]['frames']:
                    with Image.open(root/f['image']) as im:
                        if rgb_hash(im)!=f['rgb_sha256']:raise ValueError('Actual RGB differs')
                seen.add((q,rep));counts[method]+=v['correct'];method_results+=1
                exports[str(path.relative_to(root))]=sha256(path);exports[str(rawpath.relative_to(root))]=sha256(rawpath);exports[str(cp.relative_to(root))]=sha256(cp)
        if seen!={tuple(x) for x in plan['calls']}:raise ValueError('Unique call partition differs')
        for m in METHODS:
            if counts[m]!=report['metrics'][m]['correct']:raise ValueError('Summary count differs')
        calls+=len(seen);synthetic+=len(list((root/'qa'/dataset/'synthetic/calls/synthetic').glob('*.json')))
        exports[str(rp.relative_to(root))]=sha256(rp)
    if calls>780 or synthetic>8:raise ValueError('Global generation cap exceeded')
    if 'videomme' not in datasets:raise ValueError('Primary study missing')
    if read_json(root/'reports/videomme.json')['gate']['passed'] and 'mlvu' not in datasets:raise ValueError('Conditional replication is required')
    return dict(status='completed',datasets=datasets,method_results=method_results,physical_real_calls=calls,synthetic_calls=synthetic,
        mlvu_started='mlvu' in datasets,exports=exports,limitations=['observed small panels','hypothetical links','frame extraction service, not object storage','component-composed latency','no adaptive budget policy'])
