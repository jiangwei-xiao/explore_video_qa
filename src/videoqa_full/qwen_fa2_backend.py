"""2026-09-23已确认修订：只替换Qwen注意力后端，保持原视频处理与问答接口。"""
import time
from .qwen_backend import QwenVideoBackend,MIN_PIXELS,MAX_PIXELS
from .state import QWEN_PATH
from videoqa_runtime.common import offline_environment


class QwenFA2VideoBackend(QwenVideoBackend):
    """接口：继承已冻结的prepare/answer；独立加载FA2，不改SDPA历史代码或权重。"""
    def __init__(self):
        offline_environment()
        import torch
        import transformers
        import flash_attn
        from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
        # 步骤1：保持精度、随机种子与图像处理尺寸，不将FA2误称逐位等价。
        assert transformers.__version__=='4.50.0' and torch.cuda.device_count()==1
        torch.manual_seed(2027)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        self.torch=torch
        started=time.perf_counter()
        self.processor=AutoProcessor.from_pretrained(QWEN_PATH,local_files_only=True,use_fast=False,
            min_pixels=MIN_PIXELS,max_pixels=MAX_PIXELS,
            size={'shortest_edge':MIN_PIXELS,'longest_edge':MAX_PIXELS})
        # 步骤2：唯一后端差异为flash_attention_2；不回退SDPA，不改MRoPE。
        self.model=Qwen2_5_VLForConditionalGeneration.from_pretrained(QWEN_PATH,local_files_only=True,
            torch_dtype=torch.bfloat16,attn_implementation='flash_attention_2',device_map={'':'cuda:0'})
        self.model.eval().requires_grad_(False)
        assert not any(p.is_meta for p in self.model.parameters())
        assert self.model.config.vision_config.temporal_patch_size==2
        assert self.model.config._attn_implementation=='flash_attention_2'
        assert 'FlashAttention2' in type(self.model.visual.blocks[0].attn).__name__
        # 步骤3：保存实际安装版本和类名，正式新协议需单独冻结这些信息。
        self.load_report=dict(load_seconds=time.perf_counter()-started,model_path=QWEN_PATH,
            transformers=transformers.__version__,torch=torch.__version__,flash_attn=flash_attn.__version__,
            dtype=str(self.model.dtype),attention=self.model.config._attn_implementation,
            vision_attention_class=type(self.model.visual.blocks[0].attn).__name__,
            decoder_attention_class=type(self.model.model.layers[0].self_attn).__name__)
