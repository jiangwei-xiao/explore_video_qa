"""Official LLaVA loading/generation with observable input checks."""
import collections
import copy
import time
import warnings
from pathlib import Path

from .common import ROOT, SOURCE_COMMIT, offline_environment, read_json, sha256
from .protocol import PROTOCOL_VERSION, parse_answer, question_text


class LlavaBackend:
    def __init__(self):
        offline_environment()
        import subprocess
        import torch
        from llava.model.builder import load_pretrained_model

        self.torch = torch
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('Expose exactly one GPU with CUDA_VISIBLE_DEVICES')
        source = ROOT / 'third_party/LLaVA-NeXT'
        record = read_json(ROOT / 'configs/llava_source_manifest.json')
        actual_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()
        if record['commit'] != SOURCE_COMMIT or actual_commit != SOURCE_COMMIT:
            raise RuntimeError('Unexpected official source commit')
        for name, expected in record['files'].items():
            if sha256(source / name) != expected:
                raise RuntimeError(f'Official source changed: {name}')
        inventory = read_json(ROOT / 'configs/local_model_inventory.json')
        for model in inventory['models'].values():
            for name, expected in model['files'].items():
                stat = (Path(model['path']) / name).stat()
                if stat.st_size != expected['bytes'] or stat.st_mtime_ns != expected['mtime_ns']:
                    raise RuntimeError(f'Model file changed since SHA256 audit: {name}')
        torch.manual_seed(2027)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()
        begin = time.perf_counter()
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            tokenizer, model, processor, context = load_pretrained_model(
                inventory['models']['llava']['path'], None, 'llava_qwen',
                torch_dtype='bfloat16', device_map='auto', attn_implementation='sdpa',
                local_files_only=True,
            )
        self.tokenizer, self.model, self.processor, self.context = tokenizer, model, processor, context
        model.eval().requires_grad_(False)
        model.to(device='cuda:0', dtype=torch.bfloat16)
        if any(p.is_meta for p in model.parameters()):
            raise RuntimeError('Model contains unmaterialized meta parameters')
        if model.config.mm_patch_merge_type != 'spatial_unpad' or model.config.mm_newline_position != 'grid' or model.config.mm_spatial_pool_mode != 'bilinear':
            raise RuntimeError('Model visual protocol differs from the frozen configuration')
        self.load_report = {
            'load_seconds': time.perf_counter() - begin,
            'class': type(model).__name__, 'torch': torch.__version__,
            'cuda_runtime': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
            'source_commit': actual_commit, 'context_limit': context,
            'dtype': str(model.dtype), 'vision_dtype': str(model.get_vision_tower().dtype),
            'attention': model.config._attn_implementation,
            'model_inventory_sha256': sha256(ROOT / 'configs/local_model_inventory.json'),
            'loading_warnings': dict(collections.Counter(str(w.message) for w in captured)),
            'checkpoint_tensor_checks': self._check_checkpoint_tensors(inventory['models']['llava']['path']),
        }

    def _check_checkpoint_tensors(self, directory):
        from safetensors import safe_open
        torch = self.torch
        directory = Path(directory)
        weight_map = read_json(directory / 'model.safetensors.index.json')['weight_map']
        names = ['model.embed_tokens.weight', 'model.mm_projector.0.weight', 'model.image_newline']
        vision_names = sorted(n for n in weight_map if 'vision_tower' in n and n.endswith('self_attn.q_proj.weight'))
        names += vision_names[:1] + vision_names[-1:]
        if not vision_names:
            raise RuntimeError('No vision tensors found in the full LLaVA checkpoint')
        params = dict(self.model.named_parameters())
        checks = []
        for name in names:
            if name not in params or name not in weight_map:
                raise RuntimeError(f'Expected checkpoint parameter absent: {name}')
            with safe_open(str(directory / weight_map[name]), framework='pt', device='cpu') as f:
                reference = f.get_tensor(name)
                # A deterministic small slice avoids duplicating the embedding table.
                source = reference.reshape(-1)[:4096].to(dtype=params[name].dtype)
                actual = params[name].detach().reshape(-1)[:4096].cpu()
                if not torch.equal(actual, source):
                    raise RuntimeError(f'Loaded values do not match the outer checkpoint: {name}')
                checks.append({'name': name, 'elements_compared': actual.numel(), 'exact_match': True})
        return checks

    def warmup(self):
        """One synthetic full forward, never an answer-generation call."""
        from PIL import Image
        from llava.constants import IMAGE_TOKEN_INDEX
        from llava.conversation import conv_templates
        from llava.mm_utils import tokenizer_image_token
        torch = self.torch
        conversation = copy.deepcopy(conv_templates['qwen_1_5'])
        conversation.append_message(conversation.roles[0], question_text(
            'Which synthetic color is visible?', ['Gray', 'Red', 'Blue', 'Green'], 16, list(range(16))))
        conversation.append_message(conversation.roles[1], None)
        ids = tokenizer_image_token(conversation.get_prompt(), self.tokenizer, IMAGE_TOKEN_INDEX,
                                    return_tensors='pt').unsqueeze(0).cuda()
        with torch.inference_mode():
            pixels = self.processor.preprocess([Image.new('RGB', (384, 384), 'gray')] * 16,
                                               return_tensors='pt')['pixel_values'].to('cuda', torch.bfloat16)
            prepared = self.model.prepare_inputs_labels_for_multimodal(
                ids, None, torch.ones_like(ids), None, None, [pixels], ['video'])
            output = self.model(input_ids=prepared[0], position_ids=prepared[1], attention_mask=prepared[2],
                                inputs_embeds=prepared[4], use_cache=False)
            if not torch.isfinite(output.logits[:, -1, :]).all():
                raise RuntimeError('Synthetic warmup produced non-finite logits')
        torch.cuda.synchronize()

    def answer(self, frames, timestamps, duration, question, options, before_generate=None, generation_state=None):
        answer_started = time.perf_counter()
        torch = self.torch
        from llava.constants import IMAGE_TOKEN_INDEX
        from llava.conversation import conv_templates
        from llava.mm_utils import tokenizer_image_token
        from transformers import GenerationConfig

        message = question_text(question, options, duration, timestamps)
        if len(frames) != 16:
            raise ValueError('Exactly 16 input images required')
        conversation = copy.deepcopy(conv_templates['qwen_1_5'])
        conversation.append_message(conversation.roles[0], message)
        conversation.append_message(conversation.roles[1], None)
        prompt = conversation.get_prompt()
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        if int((input_ids == IMAGE_TOKEN_INDEX).sum()) != 1:
            raise ValueError('Expected one video placeholder')
        expected_visual_tokens = 16 * 14 * 15
        expected_prefill = input_ids.shape[1] - 1 + expected_visual_tokens
        context_limit = min(self.context, self.model.config.tokenizer_model_max_length)
        if expected_prefill + 8 > context_limit:
            raise ValueError('Input exceeds context; silent truncation is prohibited')
        prompt_tokenize_seconds = time.perf_counter() - answer_started
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        video = self.processor.preprocess(frames, return_tensors='pt')['pixel_values'].to('cuda:0', torch.bfloat16)
        torch.cuda.synchronize()
        preprocessing_seconds = time.perf_counter() - started
        if list(video.shape) != [16, 3, 384, 384] or not torch.isfinite(video).all():
            raise ValueError('Invalid official video preprocessing result')
        measured = {'prefill_calls': 0, 'checked_logit_steps': 0}

        def record_prefill(module, args, kwargs):
            embeddings = kwargs.get('inputs_embeds')
            if embeddings is not None and measured['prefill_calls'] == 0:
                measured['prefill_calls'] += 1
                measured['prefill_tokens'] = embeddings.shape[1]
                measured['visual_tokens'] = embeddings.shape[1] - (input_ids.shape[1] - 1)
                if embeddings.shape[1] != expected_prefill:
                    raise RuntimeError('Actual multimodal sequence was truncated or token layout changed')

        def check_logits(module, args, output):
            if not torch.isfinite(output[:, -1, :]).all():
                raise RuntimeError('Non-finite next-token logits')
            measured['checked_logit_steps'] += 1

        prefill_hook = self.model.get_model().register_forward_pre_hook(record_prefill, with_kwargs=True)
        logits_hook = self.model.lm_head.register_forward_hook(check_logits)
        config = GenerationConfig(
            max_new_tokens=8, do_sample=False, num_beams=1, use_cache=True,
            bos_token_id=self.model.generation_config.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        try:
            if before_generate:
                before_generate({'prompt': prompt, 'text_input_tokens': input_ids.shape[1],
                                 'input_token_ids': input_ids[0].tolist(),
                                 'expected_visual_tokens': expected_visual_tokens,
                                 'generation_config': config.to_dict()})
            begin = time.perf_counter()
            if generation_state is not None:
                generation_state['started'] = True
            with torch.inference_mode():
                output = self.model.generate(input_ids, images=[video], modalities=['video'],
                                             attention_mask=torch.ones_like(input_ids), generation_config=config)
            if generation_state is not None:
                generation_state['returned'] = True
            torch.cuda.synchronize()
            generation_seconds = time.perf_counter() - begin
        finally:
            prefill_hook.remove()
            logits_hook.remove()
        if measured['prefill_calls'] != 1 or measured['checked_logit_steps'] == 0:
            raise RuntimeError('Missing actual prefill or logits observations')
        parse_started = time.perf_counter()
        raw = self.tokenizer.batch_decode(output, skip_special_tokens=True)[0].strip()
        parsed = parse_answer(raw)
        answer_parse_seconds = time.perf_counter() - parse_started
        return {'protocol': PROTOCOL_VERSION, 'prompt': prompt, 'raw_output': raw,
                'parsed_answer': parsed, 'generated_token_ids': output[0].tolist(),
                'text_input_tokens': input_ids.shape[1], 'pixel_shape': list(video.shape),
                **measured, 'preprocessing_seconds': preprocessing_seconds,
                'prompt_tokenize_seconds': prompt_tokenize_seconds, 'answer_parse_seconds': answer_parse_seconds,
                'generation_seconds': generation_seconds,
                'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
                'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30}
