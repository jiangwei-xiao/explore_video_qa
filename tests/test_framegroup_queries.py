"""原词、选项顺序及标准答案隔离。"""
from videoqa_audit.framegroup_design import query_views


def test_all_option_views_ignore_answer_index():
    a=dict(question='Where is it?',options=['A. Box.','B. Bag.','C. Table.','D. Floor.'],answer_index=0)
    b=dict(a,answer_index=3)
    assert query_views(a)==query_views(b)
    assert query_views(a)[0]==('question','Where is it?')
    assert [x[0] for x in query_views(a)]==['question','option_A','option_B','option_C','option_D']
