"""事后证据审核：重建并核对实际Qwen像素Tensor，生成关键帧可见性和局部粗候选视图。"""
import argparse
from pathlib import Path
import sys
import os
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from PIL import Image,ImageDraw
from videoqa_runtime.common import read_json,sha256
from videoqa_full.state import rows,durable,utc,QWEN_PATH
from videoqa_budget32.prefix_full import BASE,DEV,METHOD
from videoqa_budget32.frames import FrameProvider
from videoqa_budget32.observe import tensor_hash

# 以下题号/时间只用于已完成实验的画面审核，不参与任何生产选帧或问答。
FOCUS={
 '004-1':{'BASE-Uniform':[3.120],'RD-1.2':[29.280],METHOD:[1.280]},
 '007-1':{'BASE-Uniform':[12.246],'RD-1.2':[12.279,13.280],METHOD:[12.279,12.779]},
 '006-3':{'BASE-Uniform':[89.690],'RD-1.2':[89.256,91.258],METHOD:[89.256,90.257]},
 '001-1':{'BASE-Uniform':[43.110],'RD-1.2':[37.771,43.277],METHOD:[43.277,48.282]},
 '602-3':{'BASE-Uniform':[2637.840],'RD-1.2':[2265.280],METHOD:[2232.280,2265.280]},
 '607-2':{'BASE-Uniform':[641.720,733.400],METHOD:[362.280]},
}


def main(run):
    """接口：仅CPU官方预处理，无视觉编码/问答；Tensor不一致就停止，不能冒充实际输入。"""
    import torch
    from transformers import AutoProcessor
    import inspect
    torch.set_num_threads(2);run=Path(run);spec=read_json(run/'protocol.json')
    processor=AutoProcessor.from_pretrained(QWEN_PATH,local_files_only=True)
    ip=processor.image_processor;assert (ip.patch_size,ip.temporal_patch_size,ip.merge_size)==(14,2,2)
    directory=run/'case_review'/'visibility';directory.mkdir(parents=True,exist_ok=True);records=[]
    # 步骤1：全部输入帧按历史顺序重建像素，先核对完整Tensor指纹再显示局部关键帧。
    for q,methods in FOCUS.items():
        for method,targets in methods.items():
            source=(DEV if q in spec['imported_ids'] else run) if method==METHOD else BASE
            result=read_json(source/'results'/method/(q+'.json'));obs=result['answer']['observed']
            images,hashes=FrameProvider(result['selection']).decode();assert hashes==result['rgb_sha256']
            data=ip(images=None,videos=[np.stack([np.asarray(im.convert('RGB')) for im in images])],size=dict(shortest_edge=100352,longest_edge=602112),return_tensors='pt')
            for im in images:im.close()
            assert tensor_hash(data['pixel_values_videos'])==obs['processor_pixel_sha256']
            assert data['video_grid_thw'].tolist()==obs['video_grid_thw']
            t,h,w=data['video_grid_thw'][0].tolist()
            # 官方布局为T,H组,W组,H合并,W合并,C,时间patch,patchH,patchW；逆排仅用于显示。
            patches=data['pixel_values_videos'].numpy().reshape(t,h//2,w//2,2,2,3,2,14,14)
            frames=patches.transpose(0,6,5,1,3,7,2,4,8).reshape(t*2,3,h*14,w*14)
            for target in targets:
                fs=result['selection']['selected_frames'];i=min(range(len(fs)),key=lambda j:abs(fs[j]['timestamp_seconds']-target))
                assert abs(fs[i]['timestamp_seconds']-target)<.003,(q,method,target)
                rgb=np.clip((frames[i].transpose(1,2,0)*np.asarray(ip.image_std)+np.asarray(ip.image_mean))*255,0,255).round().astype(np.uint8)
                p=directory/f'{q}_{method}_{fs[i]["source_pts"]}.png';Image.fromarray(rgb).save(p)
                records.append(dict(question_id=q,method=method,source_frame=fs[i],input_position=i,processed_size=obs['processed_size'],
                    verified_tensor_sha256=obs['processor_pixel_sha256'],image=str(p.relative_to(run)),image_sha256=sha256(p)))
    # 步骤2：对602-2只查看已存在的180–250秒粗候选，不密采、不运行新评分。
    q='602-2';pool=read_json(BASE/'pools'/(q+'.json'));coarse=pool['candidates'][:pool['initial_count']]
    scores=pool['initial_scores'];order=sorted(range(len(coarse)),key=lambda i:(-scores[i],coarse[i]['source_pts'],i));rank={i:k+1 for k,i in enumerate(order)}
    chosen=[f for f in coarse if 180<=f['timestamp_seconds']<=250];assert chosen
    contexts=[]
    for start in range(0,len(chosen),24):
        fs=chosen[start:start+24];images,hashes=FrameProvider(dict(video=pool['video'],selected_frames=fs)).decode()
        board=Image.new('RGB',(1280,210*((len(fs)+3)//4)),'white');draw=ImageDraw.Draw(board)
        for j,(im,f) in enumerate(zip(images,fs)):
            x,y=j%4*320,j//4*210;im.thumbnail((316,178));board.paste(im,(x,y));im.close();i=f['candidate_index']
            draw.text((x,y+180),f"{f['timestamp_seconds']:.3f}s c{i} rank{rank[i]}",fill='black')
        p=directory/f'602-2_coarse_180_250_{start//24}.jpg';board.save(p,quality=94);board.close()
        contexts.append(dict(image=str(p.relative_to(run)),frames=fs,rgb_sha256=hashes))
    durable(directory/'manifest.json',dict(status='passed',display='Inverse official patch layout and denormalization rounded to RGB; no model forward',
        processor_source_sha256=sha256(inspect.getfile(type(ip))),records=records,coarse_contexts=contexts,new_model_calls=0,utc=utc()))
    print('Verified actual pixel tensors and rendered',len(records),'views;',len(chosen),'existing coarse frames',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);main(parser.parse_args().run)
