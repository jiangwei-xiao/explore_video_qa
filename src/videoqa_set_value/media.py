"""本轮短视频顺序精确读取：避免开放GOP随机seek跳过合法目标，不改变源帧。"""
from fractions import Fraction
import hashlib
import numpy as np
from videoqa_budget32.frames import FrameProvider
from videoqa_runtime.baseline_selection import ProtocolError


class SequentialFrameProvider(FrameProvider):
    """接口：同一已选源帧集合→RGB及哈希；全片顺序解码，无重编码/重新采样。"""
    def decode(self):
        """步骤1核对时钟；步骤2按PTS和绝对源号取帧；步骤3返回与原接口同型结果。"""
        import av
        video=self.selection['video'];wanted=self.selection['selected_frames'];images=[];cursor=0;previous=None
        try:
            with av.open(video['path']) as container:
                stream=container.streams.video[0];stream.codec_context.thread_count=2
                if Fraction(stream.time_base)!=Fraction(video['time_base']) or (stream.start_time or 0)!=video['start_pts']:
                    raise ProtocolError('Source clock changed')
                for index,frame in enumerate(container.decode(stream)):
                    if frame.pts is None or (previous is not None and frame.pts<=previous):raise ProtocolError('Invalid sequential PTS')
                    previous=frame.pts;target=wanted[cursor]
                    if frame.pts>target['source_pts']:raise ProtocolError('Full decode missed exact selected PTS')
                    if frame.pts==target['source_pts']:
                        if index!=target['source_frame_index']:raise ProtocolError('Full decode source index changed')
                        images.append(frame.to_image());cursor+=1
                        if cursor==len(wanted):break
                if cursor!=len(wanted):raise ProtocolError('Selected source absent at EOF')
            hashes=[hashlib.sha256(np.ascontiguousarray(np.asarray(im.convert('RGB'))).tobytes()).hexdigest() for im in images]
            return images,hashes
        except BaseException:
            for image in images:image.close()
            raise
