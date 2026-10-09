"""exp29试点的实际前缀、OCR原词与新旧原图对照，不能据OCR自证语义正确。"""
from pathlib import Path
import sys,argparse,html
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import rows,durable
from videoqa_audit.framework_contraction import locations
from videoqa_audit.text_state_experiments import PILOT,TARGETS
from videoqa_audit.region_scope_review import draw_board


def review(root):
    """接口：只读完整池结果和父前缀，逐源验OCR RGB，输出有界复核页。"""
    root=Path(root);bank={r['question_id']:r for r in rows()};cases=[];comparisons=[]
    page=['<!doctype html><meta charset="utf-8"><title>文字状态试点</title><style>body{font:16px/1.6 sans-serif;max-width:1400px;margin:24px auto}.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}img{max-width:100%}td{padding:8px}</style><h1>文字状态试点</h1><p>原图缩略；OCR错词及时间水印可能被奖励，需要实际画面判断。尚未运行新问答。</p>']
    # 步骤1：所有8题两预算新旧图板，源像素须与用于OCR的像素完全一致。
    for q in PILOT:
        page+=['<h2>'+q+'</h2><p>'+html.escape(bank[q]['question'])+'</p><p>'+html.escape(' / '.join(bank[q]['options']))+'</p>']
        for profile,k in [('qwen32',32),('llava8',8)]:
            pp,fp,tp,sp=locations(q,k);pool=read_json(pp);before=read_json(tp);before=before['exp07'] if k==8 else before
            after_path=root/'offline'/profile/(q+'.json');after=read_json(after_path);trace=read_json(root/after['trace_file'])
            new_label=after['method'].split('-')[-1]
            for label,path in [('exp07',sp),(new_label,after_path)]:
                board=draw_board((str(root),q,profile,label,str(path)))
                if label==new_label:
                    for frame,pixel in zip(board['frames'],board['rgb_sha256']):
                        source_root=Path(after.get('ocr_source_root',str(root)))
                        record=read_json(source_root/'frame_calls'/bank[q]['video_id']/(str(frame['source_pts'])+'.json'))
                        if pixel!=record['ocr']['rgb_sha256']:raise ValueError('Selected OCR and QA source pixels differ')
            # 步骤2：精确目标只是路径诊断；其他已选同类状态另列，不直接判定语义损失。
            targets=[]
            for index in TARGETS.get(q,[]):
                a=next(r for r in before['regions'] if index in r['members']);b=next(r for r in trace['regions'] if index in r['members'])
                ka=before['region_budgets'][str(a['region_id'])]
                pos=lambda r:r['prefix'].index(index)+1 if index in r['prefix'] else None
                targets.append(dict(candidate_index=index,frame=pool['candidates'][index],tokens=after['states'][index],
                    parent_position=pos(a),new_position=pos(b),quota=ka,parent_selected=pos(a) is not None and pos(a)<=ka,
                    new_selected=index in after['selected_indices']))
            selected_words=[dict(frame=f,tokens=after['states'][f['candidate_index']]) for f in after['selection']['selected_frames']]
            item=dict(question_id=q,profile=profile,targets=targets,selected_words=selected_words,
                sources={str(p):sha256(p) for p in (pp,tp,sp,after_path)},new_qa_calls=0)
            comparisons.append(item)
            page+=['<h3>'+profile+'</h3><div class="pair">']
            for label in ('exp07',new_label):
                url=profile+'/'+q+'/'+label+'.jpg';page+=['<div>'+label+'<a href="'+url+'"><img src="'+url+'"></a></div>']
            page+=['</div><details><summary>已选原始OCR词集</summary><table>']
            for e in selected_words:page+=['<tr><td>'+str(round(e['frame']['timestamp_seconds'],3))+'</td><td>'+html.escape(' '.join(e['tokens']))+'</td></tr>']
            page+=['</table></details>']
    page+=['</html>'];(root/'review/index.html').write_text('\n'.join(page),encoding='utf-8')
    durable(root/'pilot_comparison.json',dict(cases=comparisons,selection_protocol_sha256=sha256(root/'protocol.json'),
        new_qa_calls=0,semantic_review='pending; exact source hits and OCR not automatically evidence truth'))


def main():
    """接口：独立复核显示层，不修改选择、词集或冻结问答协议。"""
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args()
    if not a.run_id.replace('_','').isalnum():p.error('Invalid run ID')
    review(ROOT/'outputs/methods'/a.run_id)


if __name__=='__main__':main()
