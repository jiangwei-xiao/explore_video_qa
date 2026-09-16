"""展开人工明确编写的事实索引范围，建立全50题新版本，不从模型答案自动生成事实。"""
import argparse
import copy
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from videoqa_runtime.common import read_json,write_json,sha256

BASE_ROLES={
 '004-3':{'neanderthal_stage':'required','archaic_stage':'partial','modern_stage':'partial'},
 '260-2':{'spray_can':'partial','toothbrush_preparation':'partial','charging':'non_target'},
 '158-3':{'score_and_time':'required','large_scorebar':'required'},
 '705-1':{'byline':'partial','name_tag':'supporting'},
 '204-3':{'mixed_posture':'supporting','female_height':'partial','male_height_readable':'required'},
 '443-1':{'game1_phonzy':'partial','game1_result':'supporting','later_attempt_marks':'non_target'},
 '316-3':{'journey_context':'supporting','letter_change':'required'},
 '870-1':{'minigolf_card':'partial','minigolf_activity':'partial','icecream':'partial'},
 '601-1':{'fan_wing_plane':'partial','biplane':'partial','invention_dispute':'required'}}


def indices(value):
    """只接受人工一基图板编号和闭区间；重复、倒序或非正编号都拒绝。"""
    result=[]
    for part in value.split(',') if value else []:
        if '-' in part:
            a,b=map(int,part.split('-'))
            if a>b:raise ValueError('倒序范围')
            result.extend(range(a,b+1))
        else:result.append(int(part))
    if any(x<1 for x in result) or len(set(result))!=len(result):raise ValueError('索引重复或非正')
    return result


def build(base,extension,out):
    """合并9题旧事实与41题新增人工事实，锁定原规格哈希，保存新版本。"""
    # 步骤1：旧规格深复制并补充人工角色声明，旧文件不改写。
    old=read_json(base);spec=copy.deepcopy(old);spec['version']='witnesses_r5'
    for qid,item in spec['questions'].items():
        for fact in item['facts']:fact['role']=BASE_ROLES[qid][fact['id']]
    # 步骤2：展开显式人工范围，不读取预测或根据正确答案筛选见证。
    for qid,item in read_json(extension).items():
        if qid in spec['questions']:raise ValueError('扩展覆盖已有题，需显式另做修订')
        facts=[]
        for fid,description,where,role,limitation in item['facts']:
            if role not in {'required','partial','supporting','non_target','uncertain'}:raise ValueError('未知事实角色')
            facts.append(dict(id=fid,description=description,union_indices=indices(where),role=role,limitation=limitation))
        spec['questions'][qid]=dict(reasoning_limits=item['limits'],facts=facts)
    if len(spec['questions'])!=50:raise ValueError('未覆盖50题')
    # 步骤3：原先空见证不等于已证明缺失；关系仍需独立判断，记录固定解释边界。
    spec['provenance']={'base_sha256':sha256(base),'extension_sha256':sha256(extension),'manual_posthoc_review':True,'new_model_calls':0}
    spec['interpretation']='必要事实/线索的人工见证目录，不是自动充分性评分；非目标事件不计答题证据。'
    if Path(out).exists() and read_json(out)!=spec:raise ValueError('已有同名规格不同，禁止原地覆盖')
    write_json(out,spec);print('compiled',len(spec['questions']),'questions',sum(len(x['facts']) for x in spec['questions'].values()),'facts')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',required=True);p.add_argument('--extension',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();build(a.base,a.extension,a.out)
