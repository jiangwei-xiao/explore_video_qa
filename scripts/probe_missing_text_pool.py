"""固定两道当前输入未定位到数值文字的问题，仅检查原粗候选池。"""
import argparse
import hashlib
from pathlib import Path

import probe_text_panel as panel
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import durable,exclusive,utc

EXPECTED={'022-1':95,'648-3':1992}


def prepare(run):
    """接口：单次顺序解码，以真实PTS和源帧号匹配2087个既有粗候选。"""
    import av
    import numpy as np
    import os
    from videoqa_audit.query_input_probe import BASE
    with exclusive(run/'prepare.lock'):
        if (run/'inputs.json').exists():
            panel.check_inputs(run)
            return
        sources={str(Path(__file__).resolve()):sha256(__file__),str(Path(panel.__file__).resolve()):sha256(panel.__file__)}
        entries=[]
        # 步骤1：只使用原1FPS池，不按OCR输出删样本，也不加入新采样时刻。
        for q,count in EXPECTED.items():
            path=BASE/'pools'/(q+'.json')
            selection_path=panel.PARENT/'selections/RD-1.2-exp12'/(q+'.json')
            pool=read_json(path);selected=set(read_json(selection_path)['selected_indices'])
            if pool['initial_count']!=count:raise ValueError('冻结粗候选数量改变')
            for p in [path,selection_path,Path(pool['video']['path'])]:sources[str(p)]=sha256(p)
            lookup={r['source_pts']:(i,r) for i,r in enumerate(pool['candidates'][:count])}
            if len(lookup)!=count:raise ValueError('粗候选源PTS不唯一')
            found=set()
            # 步骤2：从文件起点按源序号顺序解码，逐项核对，避免平均FPS推算和反复全片读取。
            with av.open(pool['video']['path']) as container:
                stream=container.streams.video[0];stream.codec_context.thread_count=2
                for index,frame in enumerate(container.decode(stream)):
                    if frame.pts not in lookup:continue
                    i,source_frame=lookup[frame.pts]
                    if index!=source_frame['source_frame_index'] or i in found:
                        raise ValueError('源帧编号或唯一性核对失败')
                    image=frame.to_image().convert('RGB')
                    output=run/'images'/f'{q}_{i}.png';output.parent.mkdir(parents=True,exist_ok=True)
                    image.save(output)
                    entries.append(dict(id=f'{q}_{i}',question_id=q,candidate_index=i,selected=i in selected,
                        diagnostic_target=False,source_frame=source_frame,image=str(output),
                        rgb_sha256=hashlib.sha256(np.asarray(image).tobytes()).hexdigest(),png_sha256=sha256(output)))
                    image.close();found.add(i)
                    if len(found)%100==0:
                        durable(run/'prepare_status.json',dict(question_id=q,completed=len(found),total=count,pid=os.getpid(),sid=os.getsid(0),utc=utc()))
                    if len(found)==count:break
            if len(found)!=count:raise ValueError('源视频没有覆盖全部冻结PTS')
        # 步骤3：冻结数据后再交给原默认OCR，模型不接收问题或目标数值。
        if len(entries)!=2087:raise ValueError('总候选分母不符')
        durable(run/'inputs.json',dict(sources=sources,entries=entries,max_images=2087,
            dataset_answers=0,fixed_questions=list(EXPECTED),
            limitation='仅两题原粗池数值文字检查；不覆盖候选间遗漏，不改变最终输入'))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--stage',choices=['prepare','recognize'],required=True)
    args=parser.parse_args()
    (prepare if args.stage=='prepare' else panel.recognize)(args.run)
