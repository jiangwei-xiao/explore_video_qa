"""固定OCR面板的选项词面出现核查；不是答案评分或最终选帧器。"""
import argparse
from pathlib import Path
import re
import sys
import unicodedata

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable


def tokens(text):
    """只规范大小写与重音，不补同义词、不根据正确选项扩展查询。"""
    text=unicodedata.normalize('NFKD',text).lower()
    return re.findall(r'[a-z]+|[0-9]+',''.join(c for c in text if not unicodedata.combining(c)))


def mentions(options,words):
    """四选项对称词面核查；保留否定词，数字须完整相同。

    字母词允许在粘连OCR词中匹配，故可能误匹配词内片段；仅作为可审计检索信号。
    不声称匹配保留语序、关系、否定语义或能辨别正确选项。
    """
    # 步骤1：沿用OCR默认0.5展示阈值；所有选项使用相同处理，不读取答案。
    visible=[t for w in words if w['score']>=0.5 for t in tokens(w['text'])]
    output=[]
    for option in options:
        option=re.sub(r'^\s*[A-D][.)]\s*','',option)
        query=[t for t in tokens(option) if t not in {'a','an','the'}]
        # 步骤2：要求全部非冠词用词出现，不把部分实体提及称为选项已出现。
        hit=bool(query) and all(any(t==v if t.isdigit() else t in v for v in visible) for t in query)
        output.append(hit)
    return output


def main(run,output):
    """接口：对比当前32帧与冻结诊断目标，不增加候选或生成任何答案。"""
    data=read_json(run/'inputs.json')
    manifest=ROOT/'data/manifests/video_mme_full_2700.json'
    questions={r['question_id']:r for r in read_json(manifest)['rows']}
    sources={str(p):sha256(p) for p in [Path(__file__),run/'inputs.json',manifest]}
    records=[]
    # 步骤1：逐图核查文字匹配，来源标签只用于事后分组，不影响匹配函数。
    for row in data['entries']:
        p=run/'calls'/(row['id']+'.json');sources[str(p)]=sha256(p)
        result=read_json(p)
        records.append(dict(id=row['id'],question_id=row['question_id'],selected=row['selected'],
            target=row['diagnostic_target'],mentions=mentions(questions[row['question_id']]['options'],result['words']),
            visible_token_count=sum(len(tokens(w['text'])) for w in result['words'] if w['score']>=0.5)))
    # 步骤2：仅输出当前未出现的选项用词，不按正确选项择优或声称新增答题证据。
    groups=[]
    for q in sorted({r['question_id'] for r in records}):
        group=[r for r in records if r['question_id']==q]
        covered=[any(r['selected'] and r['mentions'][j] for r in group) for j in range(4)]
        new=[dict(id=r['id'],new_mentions=[j for j in range(4) if r['mentions'][j] and not covered[j]])
             for r in group if not r['selected']]
        new=[r for r in new if r['new_mentions']]
        groups.append(dict(question_id=q,selected_mentions=covered,new_mentions=new))
    result=dict(sources=sources,new_model_calls=0,records=records,groups=groups,
        limitation='手工目标加入的有界面板；选项词面出现不是语义支持或答案预测，不代表全池检索能力')
    if output.exists():
        if read_json(output)!=result:raise ValueError('恢复结果变化')
    else:durable(output,result)
    for g in groups:print(g)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();main(args.run,args.output)
