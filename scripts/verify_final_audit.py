"""最终验收只读核查数据、链接、图像和诊断协议，不把文件存在当语义真值。"""
import argparse
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
import re
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from videoqa_runtime.common import read_json,write_json,sha256
from videoqa_runtime.protocol import question_text


class Links(HTMLParser):
    """解析本地静态页面中的资产/链接和帧缩略图数量，不执行网页脚本。"""
    def __init__(self):super().__init__();self.links=[];self.images=0
    def handle_starttag(self,tag,attrs):
        data=dict(attrs)
        for key in ('href','src'):
            if key in data:self.links.append(data[key])
        self.images+=tag=='img'


def verify(audit,diagnostic):
    """逐项核对原实验不变、50×6×16闭合、双尺度资产与冻结诊断；产出证据清单。"""
    from PIL import Image
    audit=Path(audit);diagnostic=Path(diagnostic);out=audit/'final_review';qs=read_json(audit/'questions.json');summary=read_json(out/'summary.json')
    # 步骤1：全量而非抽样确认清单、原始结果、每组16帧与全部图像存在/384尺寸。
    assert len(qs)==50 and len({q['video_id'] for q in qs})==50
    strata=Counter(q['stratum'] for q in qs);assert sorted(strata.values())==[16,17,17]
    hashes=read_json(audit/'protocol.json')['result_hashes'];assert len(hashes)==300
    for p,h in hashes.items():assert sha256(p)==h
    slots=0;unique=0;images=0;reviewed=0
    for q in qs:
        qid=q['question_id'];r=read_json(out/'questions'/f'{qid}.json');assert len(r['groups'])==6
        v=read_json(audit/'full_review'/'visibility'/f'{qid}.json');seen=[]
        for p in v['pages'].values():assert sha256(audit/p['board_file'])==p['sha256'];seen.extend(p['source_pts']);reviewed+=1
        assert len(seen)==len(set(seen)) and set(seen)=={f['source_pts'] for f in q['frames']}
        unique+=len(seen)
        for f in q['frames']:
            for kind in ('original','input'):
                file=audit/'frames'/qid/f'{f["source_pts"]}.{kind}.png'
                with Image.open(file) as im:
                    if kind=='input':assert im.size==(384,384)
                    im.verify()
                images+=1
        for m,g in r['groups'].items():
            assert len(g['frames'])==16 and len({f['frame']['source_pts'] for f in g['frames']})==16
            assert sum(g['counts'].values())==16
            assert Counter(f['annotation']['label'] for f in g['frames'])==Counter({k:v for k,v in g['counts'].items() if v})
            assert all(f['annotation']['reason'] and 'duplicate' in f['redundancy'] for f in g['frames']);slots+=16
    assert unique==2235 and images==4470 and slots==4800 and reviewed==165
    # 步骤2：新静态页面核对4800图像位置和所有链接，不声称真实浏览器人工视觉验收。
    link_count=0;page_images=0
    for file in [out/'index.html',*sorted((out/'pages').glob('*.html'))]:
        parser=Links();parser.feed(file.read_text());page_images+=parser.images
        for u in parser.links:
            path=u.split('#')[0]
            if path and not re.match(r'^[a-z]+:',path):assert (file.parent/path).exists(),str(file)+':'+path
            link_count+=1
    assert len(list((out/'pages').glob('*.html')))==50 and page_images==4800
    window_parser=Links();window_file=out/'source_windows.html';window_parser.feed(window_file.read_text())
    for u in window_parser.links:
        path=u.split('#')[0]
        if path:assert (window_file.parent/path).exists(),path
    # 步骤3：诊断输入应来自自身冻结原池；原/修复提示词只能因时间戳不同而改变。
    manifest=read_json(diagnostic/'manifest.json');freeze=read_json(diagnostic/'freeze.json');fp=sha256(diagnostic/'manifest.json')
    assert fp==freeze['manifest_sha256'] and len(list((diagnostic/'calls').glob('*.json')))==16
    for name,h in freeze['code_sha256'].items():assert sha256(diagnostic/'code_snapshot'/name)==h
    used_gpus=set();calls_hashes={}
    for case in manifest['cases']:
        source=read_json(case['source_result']);pool=source.get('pool',source.get('selection'));candidates={x['source_pts']:x for x in pool['candidates']}
        assert sha256(case['source_result'])==case['source_result_sha256']
        records={variant:read_json(diagnostic/'calls'/f'{case["question_id"]}.{variant}.json') for variant in ('original','repaired')}
        contents={}
        for variant,r in records.items():
            assert r['status']=='completed' and r['generation_state'].get('started') and r['generation_state'].get('returned')
            assert r['started_utc']>manifest['frozen_utc'] and r['manifest_sha256']==fp
            assert r['answer']['visual_tokens']==3360 and r['answer']['pixel_shape']==[16,3,384,384]
            assert len(r['answer']['generated_token_ids'])<=8
            frames=r['selected_frames'];assert len(frames)==16 and len({x['source_pts'] for x in frames})==16
            for x in frames:assert candidates[x['source_pts']]==x
            times=[x['timestamp_seconds'] for x in frames];assert times==sorted(times)
            contents[variant]=question_text(case['question'],case['options'],case['video']['duration_seconds'],times)
            assert contents[variant] in r['answer']['prompt'];used_gpus.add(r['gpu'])
            file=diagnostic/'calls'/f'{case["question_id"]}.{variant}.json';calls_hashes[str(file)]=sha256(file)
        assert records['original']['gpu']==records['repaired']['gpu']
        assert records['original']['answer']['prompt'].replace(contents['original'],contents['repaired'])==records['repaired']['answer']['prompt']
        assert records['original']['answer']['parsed_answer']==case['historical_prediction']
    assert len(used_gpus)<=2
    for gpu in used_gpus:
        load=read_json(diagnostic/'workers'/f'{gpu}.json')['load_report']
        assert load['dtype']==load['vision_dtype']=='torch.bfloat16' and load['attention']=='sdpa'
        assert load['source_commit']=='bce12e479bc4dfee2b9c50c88137b01ff51bd483'
        assert all(x['exact_match'] for x in load['checkpoint_tensor_checks'])
    # 步骤4：全部指定机制变化闭合，逐帧理由/事实限制和未决范围都保留。
    totals=Counter()
    for q in qs:
        r=read_json(audit/'full_review'/'semantic_changes_r6'/f'{q["question_id"]}.json')
        for key,c in r['comparisons'].items():
            totals[key+'_changed']+=c['changed'];totals[key+'_removed']+=len(c['removed']);totals[key+'_added']+=len(c['added'])
            for f in c['removed']+c['added']:assert f['initial_frame_reason'] and f['review_basis']
    assert totals['AB_changed']==33 and totals['BC_removed']==totals['BC_added']==120
    assert summary['groups']==300 and summary['errors']==128
    result={'status':'structural_protocol_acceptance_passed','questions':50,'unique_videos':50,'strata':dict(strata),'groups':300,'slots':slots,'unique_frames':unique,
            'images_verified':images,'384_boards_hashed':reviewed,'html_links_checked':link_count,'html_frame_positions':page_images,
            'source_window_images_linked':window_parser.images,'historical_hashes_checked':len(hashes),'diagnostic_started_calls':16,'diagnostic_gpus':sorted(used_gpus),'diagnostic_code_snapshot_verified':True,
            'diagnostic_prompt_only_timestamps_changed':True,'mechanism_totals':dict(totals),
            'limitations':['主观语义为事后AI初审，不是用户金标准','原片仅核查记录窗口，不是全片穷尽','未解决解释按原计划保留','静态HTML结构/链接检查，不是真实浏览器视觉验收'],
            'human_quantitative_agreement':None,'diagnostic_result_hashes':calls_hashes}
    write_json(out/'structural_acceptance.json',result);print({k:v for k,v in result.items() if k!='diagnostic_result_hashes'})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',required=True);p.add_argument('--diagnostic',required=True);a=p.parse_args();verify(a.audit,a.diagnostic)
