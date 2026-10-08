"""四个exp07全量工作点的只读共同失败复盘，新增模型调用为0。"""
from pathlib import Path
import sys,argparse,csv,json,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import rows,durable,exclusive,utc
from videoqa_audit.framework_contraction import RUN32,RUN8,RUN16,BASE
from videoqa_budget32.selection import OLD_QWEN
from videoqa_generalization.inputs import LLAVA_BASE,LLAVA_UNIFORM,DEV
from videoqa_audit.cross_configuration_failures import PROFILES,summarize_failures

METHOD='RD-1.2-exp07'


def result_path(root,method,q):
    """接口：唯一冻结结果源，兼容200题导入和早期均匀的平目录，不择优。"""
    candidates=[root/'results'/method/(q+'.json'),root/'imported/results'/method/(q+'.json')]
    if method=='BASE-Uniform':candidates.append(root/'results'/(q+'.json'))
    existing=[p for p in candidates if p.exists()]
    if len(existing)!=1:raise ValueError('Missing/ambiguous frozen result '+str((root,method,q)))
    return existing[0]


def analyze(root):
    """接口：10800份exp07与21600份基线结果逐题计分核对、哈希及分类，不生成回答。"""
    root=Path(root);root.mkdir(parents=True,exist_ok=True);data=rows()
    origins={'qwen32':RUN32,'qwen16':RUN16/'qwen16','llava8':RUN8,'llava16':RUN16/'llava16'}
    refs={'qwen32':{'BASE-Uniform':BASE,'BASE-BLIP-TopK':BASE},
          'qwen16':{'BASE-Uniform':OLD_QWEN,'BASE-BLIP-TopK':OLD_QWEN},
          'llava8':{'BASE-Uniform':RUN8,'BASE-BLIP-TopK':RUN8},
          'llava16':{'BASE-Uniform':LLAVA_UNIFORM,'BASE-BLIP-TopK':LLAVA_BASE}}
    with exclusive(root/'analysis.lock'):
        if (root/'summary.json').exists():
            old=read_json(root/'summary.json')
            for p,h in old['sources'].items():
                if sha256(p)!=h:raise ValueError('Frozen failure source changed')
            return old
        sources={str(Path(__file__)):sha256(__file__),str(ROOT/'src/videoqa_audit/cross_configuration_failures.py'):sha256(ROOT/'src/videoqa_audit/cross_configuration_failures.py'),str(DEV):sha256(DEV)}
        records={};started=time.perf_counter();counts={p:0 for p in PROFILES}
        # 步骤1：核对每个历史评价已完整验收，并绑定实际结果字节。
        for origin in origins.values():
            path=origin/'completion_audit.json'
            if read_json(path)['status']!='passed':raise ValueError('Source evaluation not accepted')
            sources[str(path)]=sha256(path)
        def load(row,profile,method,path):
            item=read_json(path);sources[str(path)]=sha256(path)
            if any(item[k]!=row[k] for k in ('question_id','question','options','video_id')):raise ValueError('Question identity differs')
            # 历史LLaVA基线使用result.answer包装；只读适配，不改旧记录或解析规则。
            answer=item['answer'] if 'answer' in item else item['result']['answer']
            pred=answer['parsed_answer'];correct=pred=='ABCD'[row['answer_index']]
            if bool(item['correct'])!=correct:raise ValueError('Independent score differs')
            s=item['selection'];frames=s['selected_frames']
            if len({x['source_pts'] for x in frames})!=len(frames) or [(x['source_pts'],x['source_frame_index']) for x in frames]!=sorted((x['source_pts'],x['source_frame_index']) for x in frames):raise ValueError('Historical frame identity invalid')
            return dict(correct=correct,prediction=pred,source_result=str(path),source_sha256=sources[str(path)],
                video=s['video'],frames=frames,rgb_sha256=item.get('rgb_sha256'),actual_frames=len(frames),
                historical_import=bool(item.get('imported',False)))
        for index,row in enumerate(data,1):
            q=row['question_id'];conditions={};baselines={}
            for profile in PROFILES:
                conditions[profile]=load(row,profile,METHOD,result_path(origins[profile],METHOD,q))
                counts[profile]+=conditions[profile]['correct'];baselines[profile]={}
                for m,path in refs[profile].items():baselines[profile][m]=load(row,profile,m,result_path(path,m,q))
            # 步骤2：同预算跨模型源帧/RGB身份不一致时拒绝称为同输入迁移。
            a,b=conditions['qwen16'],conditions['llava16'];key=lambda x:[(f['source_frame_index'],f['source_pts']) for f in x['frames']]
            if key(a)!=key(b) or a['rgb_sha256']!=b['rgb_sha256']:raise ValueError('M16 cross-model media differs')
            records[q]=dict(question_id=q,question=row['question'],options=row['options'],answer_index=row['answer_index'],
                           stratum=row['stratum'],task_type=row['task_type'],domain=row['domain'],video_id=row['video_id'],
                           conditions=conditions,baselines=baselines)
            durable(root/'records'/(q+'.json'),records[q])
            if index%30==0:durable(root/'status.json',dict(completed=index,total=2700,seconds=time.perf_counter()-started,utc=utc()))
        if counts!={'qwen32':1692,'qwen16':1629,'llava8':1610,'llava16':1680}:raise ValueError('Frozen accuracy differs')
        result=summarize_failures(data,records);dev=read_json(DEV)['rows'];dev_ids={r['question_id'] for r in dev};dev_videos={r['video_id'] for r in dev}
        result.update(correct=counts,new_model_calls=0,new_answers=0,sources=sources,seconds=time.perf_counter()-started,
            common_wrong_development=sum(q in dev_ids for q in result['common_wrong_ids']),
            common_wrong_outside_development_videos=sum(r['question_id'] in set(result['common_wrong_ids']) and r['video_id'] not in dev_videos for r in data))
        # 步骤3：导出全部共同失败原题索引；人工复核另记，不伪造全量证据充分性。
        with (root/'common_failures.csv').open('w',newline='',encoding='utf-8-sig') as stream:
            writer=csv.writer(stream);writer.writerow(['question_id','stratum','task_type','domain','question','options','gold','baseline_success_conditions'])
            for q in result['common_wrong_ids']:
                r=records[q];good=[p+'/'+m for p in PROFILES for m,x in r['baselines'][p].items() if x['correct']]
                writer.writerow([q,r['stratum'],r['task_type'],r['domain'],r['question'],json.dumps(r['options'],ensure_ascii=False),'ABCD'[r['answer_index']],';'.join(good)])
        durable(root/'summary.json',result);return result


def main():
    """接口：独立分析目录和恢复，不修改旧结果或用户未提交脚本。"""
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit-id',required=True);a=p.parse_args()
    if not a.audit_id.replace('_','').isalnum():p.error('Invalid audit ID')
    summary=analyze(ROOT/'outputs/analysis'/a.audit_id)
    print({k:summary[k] for k in ('questions','correct','common_wrong','common_wrong_video_count')},flush=True)


if __name__=='__main__':main()
