"""exp34调用边界：动态上限、来源隔离与同题完整身份。"""
from pathlib import Path
from unittest.mock import Mock
import pytest

from videoqa_audit import low_loss_development as dev
from videoqa_full.state import durable


@pytest.mark.parametrize('cell', dev.CELLS)
def test_partition_accepts_at_most_original_changed_inputs(cell):
    n, fixed_parent, maximum = dev.EXPECTED[cell]
    items = ([dict(origin='parent_import') for _ in range(fixed_parent)] +
             [dict(origin='new_generation') for _ in range(maximum)])
    assert n == len(items)
    assert dev.check_partition(*cell, items)['new_calls'] == maximum
    items[0]['origin'] = 'new_generation'
    with pytest.raises(ValueError, match='partition'):
        dev.check_partition(*cell, items)


def test_old_exp33_import_needs_no_new_call():
    cell = ('videomme', 'llava8')
    n, fixed_parent, maximum = dev.EXPECTED[cell]
    items = ([dict(origin='parent_import') for _ in range(fixed_parent)] +
             [dict(origin='exp33_import') for _ in range(3)] +
             [dict(origin='new_generation') for _ in range(maximum-3)])
    assert dev.check_partition(*cell, items) == dict(results=n,
        parent_imports=fixed_parent, exp33_imports=3, diagnostic_imports=0,
        new_calls=maximum-3)


def test_unknown_raw_return_never_generates(tmp_path, monkeypatch):
    run = tmp_path/'cell'
    durable(run/'protocol.json', dict(dataset='videomme', profile='llava8'))
    durable(run/'inputs/q1.json', dict(row={}))
    durable(run/'calls'/dev.METHOD/'q1.json', dict(
        protocol_sha256=dev.sha256(run/'protocol.json'),
        status='started_may_generate',gpu='4'))
    restore = Mock()
    monkeypatch.setattr(dev,'restore_answer',restore)
    with pytest.raises(RuntimeError, match='no retry'):
        dev.recover_call(run,'q1')
    assert not restore.called


def test_previous_result_path_has_separate_method_id():
    assert dev.METHOD == 'RD-1.2-exp34'
    assert dev.EXP33_METHOD == 'RD-1.2-exp33'
    assert dev.EXP33_DEVELOPMENT != dev.SOURCE
    assert dev.idle_worker_gpus.__name__ == 'idle_worker_gpus'
