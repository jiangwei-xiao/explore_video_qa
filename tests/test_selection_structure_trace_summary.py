"""只读结构统计的约束边界回归。"""
import importlib.util
from pathlib import Path


def module():
    path=Path(__file__).resolve().parents[1]/'scripts/summarize_selection_structure_traces.py'
    spec=importlib.util.spec_from_file_location('structure_trace_summary_test',path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def selection(ids):
    return dict(actual_frame_count=len(ids),selection=dict(selected_frames=[dict(source_frame_index=i,source_pts=i*1000) for i in ids]))


def test_source_change_can_leave_prefix_family_and_transfer_slots():
    tool=module();p=selection([0,2,3]);c=selection([1,3,4])
    parent=dict(protected_indices=[0,3],coarse_prefix_trace=dict(regions=[
        dict(region_id=0,prefix=[0,2]),dict(region_id=1,prefix=[3,4])]))
    trace=dict(posthoc_parent_mapping=dict(seed_retained=[3],anchor_retained=[3],regions=[
        dict(region_id='0',parent_quota=2,control_count=1,selected_indices=[1],selected_prefix_positions={'1':None}),
        dict(region_id='1',parent_quota=1,control_count=2,selected_indices=[3,4],selected_prefix_positions={'3':1,'4':2})]))
    value=tool.describe_control(p,c,trace,parent)
    assert not value['within_parent_feasible_family']
    assert value['outside_feasible_parent_prefix']==1 and value['transferred_region_slots']==1
    assert value['parent_overlap']==1 and value['retained_anchors']==1


def test_unchanged_region_counts_do_not_imply_same_input():
    tool=module();p=selection([0,2,3]);c=selection([1,2,3])
    parent=dict(protected_indices=[0,3],coarse_prefix_trace=dict(regions=[
        dict(region_id=0,prefix=[0,2]),dict(region_id=1,prefix=[3,4])]))
    trace=dict(posthoc_parent_mapping=dict(seed_retained=[3],anchor_retained=[3],regions=[
        dict(region_id='0',parent_quota=2,control_count=2,selected_indices=[1,2],selected_prefix_positions={'1':None,'2':2}),
        dict(region_id='1',parent_quota=1,control_count=1,selected_indices=[3],selected_prefix_positions={'3':1})]))
    value=tool.describe_control(p,c,trace,parent)
    assert value['transferred_region_slots']==0 and value['same_region_counts_different_input']
    assert not value['within_parent_feasible_family']


def test_zero_region_is_incompatible_even_when_other_frames_follow_prefix():
    tool=module();p=selection([0,1,3]);c=selection([0,1,2])
    parent=dict(protected_indices=[0,3],coarse_prefix_trace=dict(regions=[
        dict(region_id=0,prefix=[0,1,2]),dict(region_id=1,prefix=[3])]))
    trace=dict(posthoc_parent_mapping=dict(seed_retained=[0],anchor_retained=[0],regions=[
        dict(region_id='0',parent_quota=2,control_count=3,selected_indices=[0,1,2],selected_prefix_positions={'0':1,'1':2,'2':3}),
        dict(region_id='1',parent_quota=1,control_count=0,selected_indices=[],selected_prefix_positions={})]))
    value=tool.describe_control(p,c,trace,parent)
    assert value['zero_parent_regions']==1 and not value['within_parent_feasible_family']
    assert value['outside_recorded_prefix']==0 and value['outside_feasible_parent_prefix']==1
    assert tool.aggregate([])=={'n':0}
