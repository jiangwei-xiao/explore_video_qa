"""exp33零模型离线准备、冻结见证警报与有界差异复核。"""
from collections import Counter, defaultdict
import copy
from fractions import Fraction
import hashlib
import html
from pathlib import Path
import shutil
import time
import numpy as np

from videoqa_runtime.common import ROOT, read_json, sha256
from videoqa_full.state import durable, digest, exclusive, utc
from videoqa_budget32.frames import FrameProvider, validate_frames
from videoqa_methods.adjacent_context_repair import select_adjacent_repair, source_key
from videoqa_methods.coarse_eligibility import coarse_selection_cells
from videoqa_methods.density_selection import source_time

METHOD = 'RD-1.2-exp33'
PARENT = 'RD-1.2-exp12'
REVIEW = ROOT/'outputs/analysis/dual_dataset_selection_review_20261015_r2'
DIAGNOSTIC = ROOT/'outputs/diagnostics/frame_group_context_diag_20261015_r1'
FACTS = ROOT/'outputs/analysis/option_rank_dev200_20261001_r1/witness_evaluation_protocol.json'
FILES = ('src/videoqa_methods/adjacent_context_repair.py', 'src/videoqa_audit/adjacent_context_experiment.py',
         'scripts/run_adjacent_context_repair.py')
PROFILES = {'llava8': 8, 'qwen32': 32}
PANEL = ['113-3', '386-1', '103-1', '383-2', '395-2', '306-2', '170-2', '544-1', '844-1']


def publish(path, value):
    """接口：冻结导出内容不可静默覆盖，完整恢复只检查，不更改旧选择。"""
    path = Path(path)
    if path.exists():
        if read_json(path) != value: raise ValueError('Frozen output differs: '+str(path))
    else: durable(path, value)


def offline(root):
    """接口：重放9748份父输入上的固定规则，绑定历史摘要，模型调用始终为0。"""
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    with exclusive(root/'offline.lock'):
        if (root/'offline_summary.json').exists(): return verify_offline(root)
        expected = read_json(REVIEW/'sources.json')
        expected.update(read_json(REVIEW/'frozen_asset_acceptance.json')['files'])
        exports_expected = read_json(REVIEW/'summary.json')['exports']
        sources = {}; exports = {}; stats = {}; all_witnesses = []; alarms = []; began = time.perf_counter()
        code = {p:sha256(ROOT/p) for p in FILES}
        protocol = root/'protocol.json'
        if protocol.exists():
            if read_json(protocol)['code'] != code: raise ValueError('Offline code changed')
        else:
            durable(protocol, dict(method=METHOD, parent=PARENT, code=code, profiles=PROFILES,
                datasets={'videomme':2700,'mlvu':2174}, max_real_calls=820, max_synthetic_calls=8,
                policy='one retained coarse frame, direct +1/2 target, protected seeds/anchors, relative continuity, ITM-first', utc=utc()))
            for name in FILES:
                out = root/'code_snapshot'/name; out.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT/name, out)
        def bind(path, required=True):
            """步骤内读取：每个历史文件只验一次摘要，缺失原期望身份时停止。"""
            path = Path(path); key = str(path)
            if key not in sources:
                actual = sha256(path); old = expected.get(key)
                if required and old is None: raise ValueError('Frozen source hash absent: '+key)
                if old is not None and actual != old: raise ValueError('Historical source changed: '+key)
                sources[key] = actual
            return path
        facts = read_json(bind(FACTS))['entries']
        if len(facts) != 75 or len({f['question_id'] for f in facts}) != 42: raise ValueError('Fact denominator changed')
        # 步骤1：冻结开发题目身份，不重新抽样；旧清单的未授权状态不改写。
        development = read_json(bind(ROOT/'data/manifests/video_mme_development_200_20260920.json'))
        mlpath = REVIEW/'mlvu_development_210.json'
        if sha256(mlpath) != exports_expected['mlvu_development_210.json']: raise ValueError('Frozen MLVU210 changed')
        sources[str(mlpath)] = sha256(mlpath)
        subset = dict(videomme=[r['question_id'] for r in development['rows']], mlvu=[r['question_id'] for r in read_json(mlpath)['rows']])
        publish(root/'development_ids.json', subset)
        for dataset in ('videomme','mlvu'):
            for profile, budget in PROFILES.items():
                paths = sorted((REVIEW/'records'/dataset/profile).glob('*.json'))
                if len(paths) != (2700 if dataset == 'videomme' else 2174): raise ValueError('Offline population differs')
                counts = Counter(); reasons = Counter(); groups = defaultdict(Counter)
                # 步骤2：按当前池映射已有特征；K8原编号映射不能由数组长度猜测。
                for ordinal, path in enumerate(paths, 1):
                    rel = str(path.relative_to(REVIEW))
                    if sha256(path) != exports_expected[rel]: raise ValueError('Frozen parent review record changed')
                    sources[str(path)] = exports_expected[rel]
                    record = read_json(path); q = record['question_id']
                    poolpath = bind(record['pool_source']); featurepath = bind(record['feature_source']); oldpath = bind(record['answer_sources'][PARENT])
                    pool, parent = read_json(poolpath), read_json(oldpath)
                    raw = np.load(featurepath, mmap_mode='r', allow_pickle=False)
                    features = raw[pool['original_candidate_indices']] if dataset == 'videomme' and profile == 'llava8' else raw
                    index = {source_key(f):i for i,f in enumerate(pool['candidates'])}
                    selected = [index[source_key(f)] for f in parent['selection']['selected_frames']]
                    if set(selected) != set(record['structure']['selected_indices']['exp12']):
                        raise ValueError('Parent input differs from frozen selection path')
                    final, trace = select_adjacent_repair(pool, features, selected, budget)
                    selection = copy.deepcopy(parent['selection'])
                    if trace['changed']:
                        winner = trace['winner']; selection['selected_frames'][winner['replaced_slot']] = copy.deepcopy(pool['candidates'][winner['added_index']])
                    validate_frames(selection, budget)
                    cells = coarse_selection_cells(pool['plan'],pool['video'])
                    def owner(i):
                        t = source_time(pool['candidates'][i],pool['video'])
                        matches = [r['region_id'] for r in cells['regions'] if Fraction(r['left_fraction']) <= t <= Fraction(r['right_fraction'])]
                        if len(matches) != 1: raise ValueError('Candidate owner is not unique')
                        return matches[0]
                    before, after = Counter(owner(i) for i in selected), Counter(owner(i) for i in final)
                    if any(v < 1 for v in after.values()) or set(before) != set(after): raise ValueError('A protected selection unit disappeared')
                    pairs = lambda ids: [ids[j:j+2] for j in range(0,len(ids),2)]
                    item = dict(method=METHOD, parent=PARENT, dataset=dataset, profile=profile, question_id=q,
                        video_id=record['video_id'], task_type=record['task_type'], stratum=record['stratum'], selection=selection,
                        parent_result=str(oldpath), pool_source=str(poolpath), feature_source=str(featurepath),
                        source_sha256={str(p):sources[str(p)] for p in (path,poolpath,featurepath,oldpath)}, trace=trace,
                        parent_region_budgets={str(k):v for k,v in before.items()}, final_region_budgets={str(k):v for k,v in after.items()},
                        qwen_pairs_before=pairs([pool['candidates'][i]['source_pts'] for i in selected]) if profile=='qwen32' else [],
                        qwen_pairs_after=pairs([pool['candidates'][i]['source_pts'] for i in final]) if profile=='qwen32' else [])
                    output = root/'offline'/dataset/profile/(q+'.json'); publish(output,item); exports[str(output.relative_to(root))] = sha256(output)
                    counts.update(questions=1, changed=int(trace['changed']), eligible_proposals=len(trace['proposals']))
                    reasons.update(trace['reason_counts'])
                    if trace['changed']:
                        w = trace['winner']; cross = owner(w['replaced_index']) != owner(w['added_index'])
                        counts.update(cross_region=int(cross), added_fine=int(w['added_index']>=pool['initial_count']),
                            removed_fine=int(w['replaced_index']>=pool['initial_count']))
                    for label in ('task_type','stratum'):
                        groups[label+':'+str(record[label])].update(questions=1,changed=int(trace['changed']))
                    if dataset == 'videomme':
                        pts0 = {pool['candidates'][i]['source_pts'] for i in selected}; pts1 = {pool['candidates'][i]['source_pts'] for i in final}
                        for fact in (x for x in facts if x['question_id']==q):
                            oldhits = sorted(pts0 & set(fact['witness_pts'])); newhits = sorted(pts1 & set(fact['witness_pts']))
                            entry = dict(profile=profile,fact=fact,before_pts=oldhits,after_pts=newhits,
                                listed_witness_loss=bool(oldhits) and not newhits,listed_witness_gain=not oldhits and bool(newhits))
                            all_witnesses.append(entry)
                            if entry['listed_witness_loss']: alarms.append(entry)
                    if ordinal % 100 == 0:
                        durable(root/'offline_status.json',dict(status='running',dataset=dataset,profile=profile,completed=ordinal,
                            total=len(paths),seconds=time.perf_counter()-began,new_model_calls=0,utc=utc()))
                stats[dataset+'/'+profile] = dict(counts=counts,reason_counts=reasons,groups=groups)
        # 步骤3：精确对照五组人工诊断，仅评价自动化命中，不把已知答案输入算法。
        comparisons = []
        from videoqa_audit.frame_group_diagnostic import cases
        for case in cases():
            target = read_json(DIAGNOSTIC/case['id']/'inputs/T.json')['selection']
            automatic = read_json(root/'offline'/case['dataset']/case['profile']/(case['question_id']+'.json'))
            same_parent = read_json(automatic['parent_result'])['selection'] == read_json(DIAGNOSTIC/case['id']/'inputs/O.json')['selection']
            key = lambda s:[source_key(f) for f in s['selected_frames']]
            comparisons.append(dict(case=case, same_parent=same_parent, automatic_equals_diagnostic_T=key(target)==key(automatic['selection']),
                winner=automatic['trace']['winner']))
        if len(all_witnesses) != 150: raise ValueError('Two-profile 75-fact denominator changed')
        publish(root/'diagnostic_matches.json',comparisons); publish(root/'witness_paths.json',all_witnesses); publish(root/'witness_alarms.json',alarms)
        publish(root/'sources.json',sources)
        for name in ('diagnostic_matches.json','witness_paths.json','witness_alarms.json','sources.json','development_ids.json'):
            exports[name]=sha256(root/name)
        summary = dict(status='offline_complete_review_pending', questions=9748, statistics=stats,
            diagnostic_error_matches=[x['case']['id'] for x in comparisons if x['case']['id'].startswith('D') and x['same_parent'] and x['automatic_equals_diagnostic_T']],
            listed_witness_alarms=len(alarms),new_model_calls=0,sources_sha256=sha256(root/'sources.json'),
            exports=exports,protocol_sha256=sha256(protocol),seconds=time.perf_counter()-began,utc=utc())
        durable(root/'offline_summary.json',summary)
        return summary


def verify_offline(root):
    """接口：完整离线恢复只验证指纹，不重新选帧、不调用任何模型。"""
    root = Path(root); s = read_json(root/'offline_summary.json'); spec = read_json(root/'protocol.json')
    if sha256(root/'protocol.json') != s['protocol_sha256']: raise ValueError('Offline protocol changed')
    for name,h in spec['code'].items():
        if sha256(ROOT/name) != h: raise ValueError('Frozen experiment implementation changed')
    for path,h in read_json(root/'sources.json').items():
        if sha256(path) != h: raise ValueError('Historical source changed: '+path)
    for rel,h in s['exports'].items():
        if sha256(root/rel) != h: raise ValueError('Offline export changed: '+rel)
    return s


def render_review(root):
    """接口：固定十题及至多两题警报，生成差异图板；精确解码仅供审阅，不补写池。"""
    from PIL import Image, ImageDraw
    root = Path(root); out = root/'review'; out.mkdir(exist_ok=True)
    alarms = read_json(root/'witness_alarms.json')
    base = [('mlvu','mlvu-dev-Q8')]+[('videomme',q) for q in PANEL]
    extra = []
    for a in alarms:
        key = ('videomme',a['fact']['question_id'])
        if key not in base and key not in extra: extra.append(key)
        if len(extra)==2: break
    page = ['<!doctype html><meta charset="utf-8"><title>exp33固定差异复核</title>',
        '<style>body{font:16px system-ui;max-width:1550px;margin:25px auto}img{max-width:100%}pre{white-space:pre-wrap}</style>',
        '<h1>exp33：父版与单次邻接修复</h1><p>上排父输入，下排新输入；展示变更槽位前后画面。原始源图可点击。</p>']
    views = []; began = time.perf_counter()
    for dataset,q in base+extra:
        for profile in PROFILES:
            path = root/'offline'/dataset/profile/(q+'.json'); item = read_json(path); parent = read_json(item['parent_result'])
            page.append(f'<h2>{dataset}／{q}／{profile}</h2>')
            if not item['trace']['changed']:
                page.append('<p>保持父输入，无需差异复核。</p>'); views.append(dict(dataset=dataset,question_id=q,profile=profile,changed=False)); continue
            slot = item['trace']['winner']['replaced_slot']; selections = [parent['selection'], item['selection']]
            tile = Image.new('RGB',(3*470,2*305),'white'); draw = ImageDraw.Draw(tile)
            raw_files = []
            # 步骤1：只解码原始和新输入的三个槽位，不抽取新的候选或帧间插值。
            for row,selection in enumerate(selections):
                subset = dict(video=selection['video'],selected_frames=selection['selected_frames'][slot-1:min(slot+2,len(selection['selected_frames']))])
                images,_ = FrameProvider(subset).decode()
                try:
                    for col,(im,f) in enumerate(zip(images,subset['selected_frames'])):
                        dest = out/dataset/profile/q/(str(row)+'_'+str(f['source_pts'])+'.png')
                        dest.parent.mkdir(parents=True,exist_ok=True); im.save(dest); raw_files.append(str(dest.relative_to(root)))
                        thumb=im.copy();thumb.thumbnail((460,260));tile.paste(thumb,(col*470,row*305+30));thumb.close()
                        draw.text((col*470+5,row*305+5),f"{'parent' if row==0 else 'exp33'} slot={slot-1+col} t={f['timestamp_seconds']:.6f}",fill='black')
                finally:
                    for im in images:im.close()
            dest=out/dataset/profile/(q+'.jpg');dest.parent.mkdir(parents=True,exist_ok=True);tile.save(dest,quality=93);tile.close()
            rel=dest.relative_to(out);page += [f'<a href="{rel}"><img src="{rel}"></a>',
                '<pre>'+html.escape(str(item['trace']['winner']))+'</pre>',
                f'<a href="../offline/{dataset}/{profile}/{q}.json">完整选择与配额</a>']
            views.append(dict(dataset=dataset,question_id=q,profile=profile,changed=True,contact_sheet=str(dest.relative_to(root)),raw_files=raw_files))
    (out/'index.html').write_text('\n'.join(page))
    durable(out/'manifest.json',dict(questions=base+extra,views=views,max_questions=12,render_seconds=time.perf_counter()-began,utc=utc()))
    return views
