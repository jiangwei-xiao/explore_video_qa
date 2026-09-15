"""审计验收与快照：核查原始结果未变、派生产物完整和明确的待审核状态。"""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from videoqa_audit.core import read,write,digest,validate_annotation


def review_progress(counts):
    """根据实际初审计数生成状态；不将首批40题缺项写死在可复用验收器中。"""
    pending=counts['pending_questions']
    incomplete=([f'其余{pending}题逐帧初审'] if pending else [])
    incomplete += ['全部384可见性核查状态需另查','用户复核与一致率状态需另查','全量机制结论与两条建议最终判定需另查']
    return ('initial_review_incomplete' if pending else 'initial_review_complete_other_checks_pending'),incomplete


def verify(out):
    """验证整个审计目录并保存代码/依赖快照；不修改旧实验或Git索引。"""
    from PIL import Image
    out=Path(out);repo=Path(__file__).resolve().parents[1]
    protocol=read(out/'protocol.json');questions=read(out/'questions.json')
    # 步骤1：300条历史问答逐文件核对；不对不一致资产继续出报告。
    for path,expected in protocol['result_hashes'].items():
        if digest(path)!=expected:raise ValueError(f'历史输入已变: {path}')
    counts=dict(reviewed_questions=0,reviewed_unique_frames=0,reviewed_slots=0,pending_questions=0)
    images=0
    for q in questions:
        a=read(out/'annotations'/f'{q["question_id"]}.json');validate_annotation(q,a)
        root=out/'frames'/q['question_id']
        if not (root/'complete.json').exists():raise ValueError('视图未完成')
        for row in q['frames']:
            for kind in ('original','input'):
                with Image.open(root/f'{row["source_pts"]}.{kind}.png') as image:
                    if kind=='input' and image.size!=(384,384):raise ValueError('输入视图尺寸不符')
                    image.verify();images+=1
        if a['status']=='initial_reviewed':
            counts['reviewed_questions']+=1;counts['reviewed_unique_frames']+=len(q['frames']);counts['reviewed_slots']+=96
        else:counts['pending_questions']+=1
    groups=read(out/'per_group.json')
    if len(groups)!=300 or any(sum(g['counts'].values())!=16 for g in groups):raise ValueError('组级数量不闭合')
    if len(read(out/'bc_replacements.json'))!=240 or len(read(out/'ab_adaptation.json'))!=33:raise ValueError('机制集合差数量不符')
    # 步骤2：快照只含本轮代码/配置及依赖锁，不复制模型/凭据或用户无关文件。
    files=list((repo/'src/videoqa_audit').glob('*.py'))+[repo/'src/videoqa_audit/review.html']
    files += [repo/'scripts'/name for name in ('audit_evidence.py','render_evidence_context.py','build_input_boards.py','verify_evidence_audit.py')]
    files += [repo/'tests/test_evidence_audit.py',repo/'configs/evidence_requirements_v1.json',repo/'configs/evidence_pilot_windows.json']
    files += list(repo.glob('requirements*.txt'))
    snapshots={}
    for path in files:
        relative=path.relative_to(repo);target=out/'code_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target);snapshots[str(relative)]=digest(target)
    write(out/'code_snapshot_manifest.json',snapshots)
    # 步骤3：实际执行回归测试并保留原始日志；不把固定数字当作当前测试结果。
    tests=subprocess.run([sys.executable,'-m','pytest','-q'],cwd=repo,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (out/'tests.log').write_text(tests.stdout,encoding='utf-8')
    if tests.returncode:raise ValueError('测试未通过，详见tests.log')
    def number(word):
        """从真实pytest总结读取数量；未出现该项时返回0。"""
        match=re.search(r'(\d+) '+word,tests.stdout);return int(match.group(1)) if match else 0
    # 步骤4：区分已完成工程与待完成研究；真实浏览器和用户一致率不伪造通过。
    status,incomplete=review_progress(counts)
    result=dict(historical_results_verified=300,images_verified=images,groups_verified=300,**counts,
                final_qa_calls=0,scope_calls=0,scoring_calls=0,
                test_suite=dict(passed=number('passed'),skipped=number('skipped'),warnings=number('warning'),log_sha256=digest(out/'tests.log')),
                ui_test=read(out/'ui_smoke.json') if (out/'ui_smoke.json').exists() else {'status':'pending'},
                git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
                staged_diff_sha256=hashlib.sha256(subprocess.check_output(['git','diff','--cached','--binary'],cwd=repo)).hexdigest(),
                status=status,incomplete=incomplete)
    write(out/'verification.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);verify(p.parse_args().out)
