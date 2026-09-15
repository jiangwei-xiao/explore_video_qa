"""事后分析补查连带替换与λ作用路径，不产生模型调用或新实验版本。"""
import argparse
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from videoqa_runtime.common import ROOT,read_json,write_json


def analyze(run):
    """接口：读取已完成A/B/C/D结果，量化配额改变、窗口外旧帧替换和分类分组得失。"""
    # 步骤1：按冻结题序读取结果，不改变任何选择或预测。
    run=Path(run); rows=read_json(ROOT/'data/manifests/video_mme_development.json')['rows']
    details=[]; scope_groups={}
    for row in rows:
        qid=row['question_id']; r={v:read_json(run/'results'/v/f'{qid}.json') for v in 'ABCD'}
        pools={v:r[v]['pool'] for v in 'ABCD'}
        sets={v:{x['source_pts']:x for x in pools[v]['selected_frames']} for v in 'ABCD'}
        tb=Fraction(pools['C']['video']['time_base']); origin=pools['C']['video']['start_pts']
        windows=[(Fraction(h['left_fraction']),Fraction(h['right_fraction'])) for h in pools['C']['hotspot_trace'] if h['selected']]
        # 步骤2：区分窗口内替换与整段重选导致的窗口外连带替换；距离按精确PTS计算。
        removed=[]
        for pts in sorted(sets['B'].keys()-sets['C'].keys()):
            time=(pts-origin)*tb
            distance=min((max(left-time,time-right,Fraction(0)) for left,right in windows),default=None)
            removed.append(dict(source_pts=pts,time_seconds=float(time),
                                distance_to_nearest_window_seconds=None if distance is None else float(distance)))
        old_added=[x for pts,x in sets['C'].items() if pts not in sets['B'] and x['candidate_index']<pools['C']['initial_count']]
        changed_quota=pools['A']['quotas']!=pools['B']['quotas']
        scope=pools['B']['scope']['label']
        group=scope_groups.setdefault(scope,dict(n=0,selection_changed=0,quota_changed=0,improved=0,regressed=0))
        group['n']+=1; group['selection_changed']+=sets['A'].keys()!=sets['B'].keys(); group['quota_changed']+=changed_quota
        group['improved']+=r['B']['correct'] and not r['A']['correct']; group['regressed']+=r['A']['correct'] and not r['B']['correct']
        details.append(dict(question_id=qid,scope=scope,A_to_B_quota_changed=changed_quota,
                            A_to_B_selection_changed=sets['A'].keys()!=sets['B'].keys(),
                            B_to_C_selection_changed=sets['B'].keys()!=sets['C'].keys(),removed_coarse_frames=removed,
                            newly_selected_old_candidates=[x['timestamp_seconds'] for x in old_added],
                            new_fine_selected=sum(x['candidate_index']>=pools['C']['initial_count'] for x in sets['C'].values()),
                            outside_removed=sum(x['distance_to_nearest_window_seconds'] is not None and x['distance_to_nearest_window_seconds']>0 for x in removed),
                            far_outside_removed=sum(x['distance_to_nearest_window_seconds'] is not None and x['distance_to_nearest_window_seconds']>1 for x in removed),
                            correct={v:r[v]['correct'] for v in 'ABCD'}))
    # 步骤3：一秒仅作为事后区分边界舍入与明显非局部替换的描述尺度，不是方法参数。
    report=dict(scope_groups=scope_groups,
                B_to_C_changed_questions=sum(x['B_to_C_selection_changed'] for x in details),
                total_removed_coarse=sum(len(x['removed_coarse_frames']) for x in details),
                total_new_fine_selected=sum(x['new_fine_selected'] for x in details),
                total_newly_selected_old_candidates=sum(len(x['newly_selected_old_candidates']) for x in details),
                questions_with_outside_removal=sum(x['outside_removed']>0 for x in details),
                total_outside_removals=sum(x['outside_removed'] for x in details),
                questions_with_removal_more_than_1s_outside=sum(x['far_outside_removed']>0 for x in details),
                total_removals_more_than_1s_outside=sum(x['far_outside_removed'] for x in details),
                details=details,note='仅事后诊断；不改样本、参数或预测；1秒是粗采样间隔对应的描述尺度。')
    write_json(run/'cascade_analysis.json',report)
    print({k:v for k,v in report.items() if k!='details'})
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='分析补查的窗口外连带替换，不调用模型')
    parser.add_argument('run_dir',type=Path)
    analyze(parser.parse_args().run_dir)
