"""有界评分器案例面板：固定顺序、配对得失、不扩审计或伪造语义初审。"""
from pathlib import Path
import pytest
from videoqa_full.state import durable
from videoqa_audit import scorer_comparison_review as review


def test_panel_requires_complete_comparison(tmp_path):
    """接口：不能根据中途分数挑补充案例。"""
    with pytest.raises(ValueError,match='Complete paired'):
        review.freeze_panel(tmp_path)


def test_fixed_panel_and_first_gains_losses_are_bounded(tmp_path,monkeypatch):
    """接口：先固定六题，再按冻结顺序交替取改善/退化，最多12题且保持待初审。"""
    durable(tmp_path/'comparison.json',{'status':'completed'})
    rows=[dict(question_id=f'{i:03d}-1') for i in range(1,25)]
    monkeypatch.setattr(review,'rows',lambda:rows)
    monkeypatch.setattr(review,'parent_result',lambda p,q,m:Path('/historical')/p/m/(q+'.json'))
    original=review.read_json
    def reader(path):
        path=Path(path)
        if path.suffix=='.json' and path.stem[:3].isdigit():
            even=int(path.stem[:3])%2==0
            return {'correct':even if 'historical' in path.parts else not even}
        return original(path)
    monkeypatch.setattr(review,'read_json',reader)
    result=review.freeze_panel(tmp_path)
    assert result['question_ids']==list(review.FIXED)+[f'{i:03d}-1' for i in range(1,7)]
    assert len(result['supplemental_slots'])==6 and result['maximum_questions']==12
    assert result['new_model_calls']==0 and result['semantic_status']=='pending_actual_frame_review'
