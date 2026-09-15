"""从视频起点独立解码，核对局部补查记录的绝对源帧号；不调用模型。"""
import argparse
import concurrent.futures
import json
import time
from pathlib import Path


def verify_one(path):
    """接口：读取一题C结果，从文件开头计数解码，返回粗候选及新增候选的帧号核验数量。"""
    # 步骤1：C包含所有初始及新增候选，其他变体的候选一致性已由机制审计确认。
    import av
    record=json.loads(Path(path).read_text()); pool=record['pool']
    expected={r['source_pts']:r['source_frame_index'] for r in pool['candidates']}
    remaining=dict(expected); total=0; started=time.perf_counter()
    # 步骤2：不使用局部seek或粗锚点，从视频起点独立累计绝对帧号。
    with av.open(pool['video']['path']) as container:
        stream=container.streams.video[0]; stream.codec_context.thread_count=2
        for index,frame in enumerate(container.decode(stream)):
            total=index+1
            if frame.pts in remaining:
                assert index==remaining.pop(frame.pts),(record['question_id'],frame.pts,index,expected[frame.pts])
            if not remaining: break
    assert not remaining,(record['question_id'],remaining)
    return dict(question_id=record['question_id'],checked=len(expected),initial=pool['initial_count'],
                fine=len(expected)-pool['initial_count'],decoded_frames=total,seconds=time.perf_counter()-started)


def verify(run):
    """核心接口：并行核对50个视频，保存独立源帧身份报告；无任何问答或分类调用。"""
    # 步骤1：固定要核验的50个已完成结果，最多8个CPU任务并行，每任务解码2线程。
    run=Path(run); paths=sorted((run/'results/C').glob('*.json'))
    assert len(paths)==50
    started=time.perf_counter(); results=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as pool:
        for result in pool.map(verify_one,paths):
            results.append(result)
            print(f'Source frame audit: {len(results)}/50',flush=True)
    # 步骤2：只输出审计文件，原始结果和执行指纹保持不变。
    report=dict(status='passed',videos=50,checked=sum(r['checked'] for r in results),
                initial=sum(r['initial'] for r in results),fine=sum(r['fine'] for r in results),
                seconds=time.perf_counter()-started,qa_calls=0,scope_calls=0,results=results)
    destination=run/'source_frame_identity_audit.json'
    with destination.open('x') as f:
        json.dump(report,f,ensure_ascii=False,indent=2); f.write('\n')
    print({k:v for k,v in report.items() if k!='results'},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='独立核对所有粗细候选的真实源帧号')
    parser.add_argument('run_dir',type=Path)
    verify(parser.parse_args().run_dir)
