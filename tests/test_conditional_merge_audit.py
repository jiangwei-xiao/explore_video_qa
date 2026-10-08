"""独立图连通审计与生产合并实现使用不同计算路径。"""
from scripts.audit_conditional_merge_completion import expected_groups


def test_sparse_bridge_does_not_connect_dense_components():
    cores=[dict(core_id=0,seed_count=3,left_fraction='0',right_fraction='6'),
           dict(core_id=1,seed_count=1,left_fraction='5',right_fraction='12'),
           dict(core_id=2,seed_count=3,left_fraction='11',right_fraction='18')]
    assert expected_groups(cores)==[[0],[1],[2]]


def test_touching_dense_components_connect_around_sparse_core():
    cores=[dict(core_id=0,seed_count=3,left_fraction='0',right_fraction='10'),
           dict(core_id=1,seed_count=1,left_fraction='7',right_fraction='15'),
           dict(core_id=2,seed_count=3,left_fraction='10',right_fraction='20')]
    assert expected_groups(cores)==[[0,2],[1]]
