"""将核查确认的82秒无字问题修订为新事实版本，旧版不改写。"""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from videoqa_runtime.common import read_json,write_json,sha256


def revise(audit):
    """输入审计根目录，保留r5并生成r6及明确修订理由。"""
    audit=Path(audit);old=audit/'full_review'/'witness_spec_r5.json';spec=read_json(old)
    # 步骤1：只修订已实际重看确认的事实，不依答案调整规则。
    item=spec['questions']['004-3'];facts=item['facts']
    f=next(f for f in facts if f['id']=='archaic_stage');f['union_indices']=[40,41,42]
    facts.append(dict(id='unlabeled_transition',description='82秒无字过渡身体形态',union_indices=[43],role='uncertain',limitation='没有Archaic名称，不能由审核者原片上下文补进模型输入。'))
    # 步骤2：补齐同一演化阶段的替代见证，避免其余换帧因目录过粗无法解释。
    for fid,desc,ids in [('dryopithecus','Dryopithecus',[1,2,3,4]),('ramapithecus','Ramapithecus',[5,6,7,8]),('ardipithecus','Ardipithecus',[9,10,11]),('anamensis','Anamensis',[12,13,14]),('afarensis','Afarensis',[15,16,17,18]),('robustus','Robustus',[19,20]),('boisei','Boisei',[21,22]),('habilis','Habilis',[23,24,25,26,27,28]),('erectus','Erectus',[30,31,32,33,34,35,36,37,38,39])]:
        facts.append(dict(id=fid,description=desc+'阶段比较背景',union_indices=ids,role='supporting',limitation='比较背景，不独自证明目标阶段快速退毛；淡出文字的可靠性依384记录。'))
    # 步骤3：新版本留存输入哈希与更正，不覆盖既有事实或统计报告。
    spec['version']='witnesses_r6';spec['revision']={'previous_sha256':sha256(old),'reason':'82.28秒无阶段文字；撤销其作为Archaic名称见证，并补充其他阶段的替代背景见证。','rechecked_input':'frames/004-3/1053184.input.png','input_sha256':sha256(audit/'frames/004-3/1053184.input.png')}
    target=audit/'full_review'/'witness_spec_r6.json'
    if target.exists() and read_json(target)!=spec:raise ValueError('r6已存在不同内容，需新版本')
    write_json(target,spec);print('r6 preserved prior version and corrected frame 82.28')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',required=True);a=p.parse_args();revise(a.audit)
