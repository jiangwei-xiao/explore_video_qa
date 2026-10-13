"""冻结检索前端：官方ITM与纯视觉CLS共同返回，不引入语言生成。"""
import time
import numpy as np

REVISION = '520bf73fd0ef6ccce791cd3f06908040aef2d8cc'
MODEL_ID = 'Salesforce/blip2-itm-vit-g'
WEIGHT_SHA256 = '31b816e89e6d0a398ffcc2403f923a07e3b7e3f684e7b5199a986a8f355bed12'


def normalize_cls(values):
    """接口：二维有限纯视觉CLS→FP32单位特征；零向量不能静默归一化。"""
    x = np.asarray(values, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError('CLS must be a matrix')
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    if x.ndim != 2 or not np.isfinite(x).all() or np.any(norm <= 1e-12):
        raise ValueError('Invalid pure-visual CLS')
    return x / norm


class Blip2ItmFrontend:
    """接口：兼容既有粗扫描/细帧编码契约，前端只接收原问题与RGB。"""
    def __init__(self, path):
        """加载固定本地检索权重，FP32/batch16，不执行生成或网络下载。"""
        import torch
        from transformers import Blip2ForImageTextRetrieval, Blip2Processor
        from videoqa_budget32.selection import configure_visual_precision
        self.torch = torch
        configure_visual_precision(torch)
        self.processor = Blip2Processor.from_pretrained(path, local_files_only=True)
        self.model = Blip2ForImageTextRetrieval.from_pretrained(
            path, local_files_only=True, torch_dtype=torch.float32).eval().requires_grad_(False).cuda()
        self.limit = self.model.config.qformer_config.max_position_embeddings
        self.counts = dict(itm_batches=0, itm_frames=0, visual_batches=0, visual_frames=0)

    def tokenize(self, question):
        """接口：完整原问题编码，显式拒绝超长；不继承LAVIS的32Token截断。"""
        value = self.processor.tokenizer(question, truncation=False, return_tensors='pt')
        if value['input_ids'].shape[1] > self.limit:
            raise ValueError('Question exceeds actual QFormer text position limit')
        return value

    def score_batch_features(self, encoded, images):
        """接口：一次官方ITM前向→正类概率/纯视觉CLS/同步预处理与前向耗时。"""
        t = self.torch
        if not 1 <= len(images) <= 16:
            raise ValueError('Frozen batch16 exceeded')
        # 步骤1：使用固定revision官方图像处理器；纯视觉与文本输入分开构造。
        t.cuda.synchronize(); start = time.perf_counter()
        pixels = self.processor(images=images, return_tensors='pt')['pixel_values'].to('cuda', t.float32)
        inputs = {k: v.repeat(len(images), 1).cuda() for k, v in encoded.items()
                  if k in ('input_ids', 'attention_mask')}
        t.cuda.synchronize(); pre = time.perf_counter() - start; start = time.perf_counter()
        self.counts['itm_batches'] += 1; self.counts['itm_frames'] += len(images)
        # 步骤2：显式ITM分支；视觉字段在QFormer交互之前，不取ITC投影输出。
        with t.inference_mode():
            out = self.model(pixel_values=pixels, **inputs, use_image_text_matching_head=True,
                             return_dict=True)
            scores = out.logits_per_image.float().softmax(-1)[:, 1].cpu().tolist()
            cls = out.vision_model_output.last_hidden_state[:, 0, :].float()
            features = normalize_cls(cls.cpu().numpy())
        t.cuda.synchronize(); forward = time.perf_counter() - start
        if not np.isfinite(scores).all() or any(not 0 <= v <= 1 for v in scores):
            raise ValueError('Invalid ITM positive probabilities')
        return scores, features, pre, forward

    def visual_batch(self, images):
        """接口：补查帧只经同一纯视觉编码器，无文本、无ITM和语言生成。"""
        t = self.torch
        if not 1 <= len(images) <= 16:
            raise ValueError('Frozen visual batch16 exceeded')
        # 步骤1：与粗帧一致的图像处理和FP32；同步墙钟覆盖异步GPU工作。
        t.cuda.synchronize(); start = time.perf_counter()
        pixels = self.processor(images=images, return_tensors='pt')['pixel_values'].to('cuda', t.float32)
        t.cuda.synchronize(); pre = time.perf_counter() - start; start = time.perf_counter()
        self.counts['visual_batches'] += 1; self.counts['visual_frames'] += len(images)
        # 步骤2：直接调用vision_model，不使用QFormer；特征维度由实际模型输出决定。
        with t.inference_mode():
            cls = self.model.vision_model(pixel_values=pixels, return_dict=True).last_hidden_state[:, 0, :]
            features = normalize_cls(cls.float().cpu().numpy())
        t.cuda.synchronize()
        return features, pre, time.perf_counter() - start

    def synthetic_acceptance(self):
        """接口：合成batch16及末批1验证文本不污染CLS、视觉路径一致与概率有限。"""
        from PIL import Image
        images = [Image.new('RGB', (896, 672), (20 + i * 7, 40, 100)) for i in range(16)]
        try:
            scores, a, _, _ = self.score_batch_features(self.tokenize('Which color is visible?'), images)
            _, b, _, _ = self.score_batch_features(self.tokenize('Is the background blue?'), images)
            c, _, _ = self.visual_batch(images)
            self.score_batch_features(self.tokenize('Which color is visible?'), images[:1])
            error = max(float(np.max(np.abs(a-b))), float(np.max(np.abs(a-c))))
            if error > 1e-6:
                raise ValueError('Official pure-vision paths differ')
            return dict(status='passed', absolute_error=error, feature_dim=a.shape[1],
                        scores=scores, preprocessing=self.processor.image_processor.to_dict(),
                        text_position_limit=self.limit, counts=dict(self.counts), synthetic_only=True)
        finally:
            for im in images: im.close()
