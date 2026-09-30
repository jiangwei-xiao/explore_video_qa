"""全量exp07有界案例图板：按冻结规则选题，精确取帧，不调用问答或评分模型。"""
import argparse
import html
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from PIL import Image, ImageDraw
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable,rows
from videoqa_budget32.prefix_full import BASE,DEV,METHOD
from videoqa_budget32.frames import FrameProvider


def render(job):
    """接口：输出完整输入图板及相对exp07的差异标记，不把未显示帧当作缺失证据。"""
    run,q,method = job
    source = (DEV if q in read_json(run/'protocol.json')['imported_ids'] else run) if method == METHOD else BASE
    value = read_json(source/'results'/method/(q+'.json'))
    new_source = DEV if q in read_json(run/'protocol.json')['imported_ids'] else run
    latest = read_json(new_source/'results'/METHOD/(q+'.json'))
    key = lambda f:(f['source_pts'],f['source_frame_index'])
    shared = {key(f) for f in latest['selection']['selected_frames']}
    frames = value['selection']['selected_frames']; images,hashes = FrameProvider(value['selection']).decode()
    assert hashes == value['rgb_sha256']
    directory = run/'case_review'/q; directory.mkdir(parents=True,exist_ok=True)
    board = Image.new('RGB',(1280,210*((len(images)+3)//4)),'white'); paint = ImageDraw.Draw(board)
    # 步骤1：展示实际全部32帧；原比例缩略不冒充Qwen预处理的可见性结论。
    for i,(image,frame) in enumerate(zip(images,frames)):
        x,y = i%4*320,i//4*210
        image.thumbnail((316,178)); board.paste(image,(x,y)); image.close()
        label = f"{i+1}: {frame['timestamp_seconds']:.3f}s"
        if method != METHOD: label += ' KEEP' if key(frame) in shared else ' OUT'
        paint.text((x,y+180),label,fill='black')
    path = directory/(method+'.jpg'); board.save(path,quality=92); board.close()
    durable(directory/(method+'.json'),dict(question_id=q,method=method,selection=value['selection'],rgb_sha256=hashes,
        answer=value['answer']['parsed_answer'],raw_output=value['answer']['raw_output'],correct=value['correct'],
        source=str(source),image_sha256=sha256(path),new_model_calls=0))


def main(run):
    """接口：最多12题三组完整画面，结论留给限时人工初审，未审核不伪装不确定。"""
    run = Path(run); manifest = read_json(run/'case_manifest.json'); qs = manifest['unique_ids']; assert len(qs)<=12
    methods = ['BASE-Uniform','RD-1.2',METHOD]
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(render,[(run,q,m) for q in qs for m in methods]))
    data = {r['question_id']:r for r in rows()}
    page = ['<!doctype html><meta charset="utf-8"><title>exp07全量有界案例复核</title>',
        '<h1>exp07全量有界案例复核</h1><p>固定题序首个分层改善与退化，事后审核。完整输入展示；KEEP/OUT指旧输入帧是否被exp07保留，不代表事实得失。原比例缩略图不等于Qwen实际预处理图。</p>']
    for q in qs:
        r = data[q]
        page += [f'<h2>{q}</h2><p>{html.escape(r["question"])}</p>',f'<p>{html.escape(" / ".join(r["options"]))}</p>',f'<p>标准答案：{"ABCD"[r["answer_index"]]}</p>']
        for m in methods:
            result = read_json(run/'case_review'/q/(m+'.json'))
            page += [f'<h3>{m}：{result["answer"]}，正确={result["correct"]}</h3>',f'<a href="{q}/{m}.jpg"><img style="max-width:100%" src="{q}/{m}.jpg"></a>']
    (run/'case_review/index.html').write_text('\n'.join(page),encoding='utf-8')
    durable(run/'case_review/review_status.json',dict(status='pending_review',question_ids=qs,new_model_calls=0))
    print('rendered',len(qs),'cases',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--run',required=True)
    main(parser.parse_args().run)
