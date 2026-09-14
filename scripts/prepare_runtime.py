"""Freeze official source, local model bytes, and an offline SigLIP alias."""
import datetime
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from videoqa_runtime.common import ROOT, SOURCE_COMMIT, read_json, sha256, write_json

MODELS = {
    'llava': Path('/home/models/lmms-lab/LLaVA-Video-7B-Qwen2'),
    'vision': Path('/home/models/google/siglip-so400m-patch14-384'),
    'blip': Path('/home/models/Salesforce/blip-itm-base-coco'),
}


def main():
    source = ROOT / 'third_party/LLaVA-NeXT'
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()
    if commit != SOURCE_COMMIT or subprocess.check_output(['git', 'diff', 'HEAD', '--'], cwd=source):
        raise RuntimeError('Official checkout does not match the frozen source commit')
    names = subprocess.check_output(['git', 'ls-files', 'llava', 'LICENSE', 'pyproject.toml', 'docs/LLaVA_Video_1003.md'], cwd=source, text=True).splitlines()
    source_record = {'url': 'https://github.com/LLaVA-VL/LLaVA-NeXT', 'commit': commit,
                     'modified_upstream_files': [], 'files': {name: sha256(source / name) for name in names}}
    write_json(ROOT / 'configs/llava_source_manifest.json', source_record)
    models = {}
    for name, directory in MODELS.items():
        paths = sorted(p for p in directory.iterdir() if p.is_file() and (
            p.suffix in ('.json', '.safetensors', '.txt', '.model') or p.name == 'pytorch_model.bin'
        ) and p.name != 'trainer_state.json')
        if not paths or not (directory / 'config.json').is_file():
            raise FileNotFoundError(directory)
        files = {}
        for path in paths:
            stat = path.stat()
            files[path.name] = {'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns, 'sha256': sha256(path)}
        if name == 'llava':
            index = read_json(directory / 'model.safetensors.index.json')
            if not set(index['weight_map'].values()).issubset(files):
                raise RuntimeError('Missing LLaVA checkpoint shard')
        required_weights = {'vision': 'model.safetensors', 'blip': 'pytorch_model.bin'}
        if name in required_weights and required_weights[name] not in files:
            raise FileNotFoundError(required_weights[name])
        models[name] = {'path': str(directory), 'files': files}
        print(f'Hashed {name}: {len(files)} files', flush=True)

    # This is a content-derived local cache key, NOT an upstream commit claim.
    import hashlib
    identity = hashlib.sha256(json.dumps(models['vision']['files'], sort_keys=True).encode()).hexdigest()[:40]
    cache = ROOT / '.cache/huggingface/hub/models--google--siglip-so400m-patch14-384'
    snapshot = cache / 'snapshots' / identity
    snapshot.mkdir(parents=True, exist_ok=True)
    for filename in models['vision']['files']:
        target = snapshot / filename
        original = MODELS['vision'] / filename
        if target.is_symlink() and target.resolve() == original.resolve():
            continue
        if target.exists() or target.is_symlink():
            raise FileExistsError(target)
        target.symlink_to(original)
    (cache / 'refs').mkdir(exist_ok=True)
    # HF's local snapshot resolver expects the exact key, without a newline.
    (cache / 'refs/main').write_text(identity)
    inventory = {'checked_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 'identity_scope': 'SHA256 of local files; upstream LFS identity not verified',
                 'models': models, 'siglip_local_cache_key': identity,
                 'siglip_repo_id': 'google/siglip-so400m-patch14-384'}
    write_json(ROOT / 'configs/local_model_inventory.json', inventory)
    versions = {dist.metadata['Name']: dist.version for dist in importlib.metadata.distributions()}
    write_json(ROOT / 'configs/runtime_environment.json', {
        'python': sys.version, 'executable': sys.executable,
        'prefix': sys.prefix, 'environment_type': 'conda', 'packages': versions})
    print('Source, model inventory, and offline SigLIP cache ready.', flush=True)


if __name__ == '__main__':
    main()
