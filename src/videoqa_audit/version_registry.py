"""统一方法名称查询；仅做元数据映射，不修改冻结实验的内部字段。"""
import json
from pathlib import Path

REGISTRY = Path(__file__).resolve().parents[2] / 'configs/method_versions.json'

def load_registry(path=None):
    """接口：读取唯一机器登记源，供新报告/工具引用。"""
    return json.loads(Path(path or REGISTRY).read_text(encoding='utf-8'))

def canonical_name(value, scope=None, registry=None):
    """接口：旧别名转统一ID；A/E1/v1等必须给出作用域，禁止猜测答案字母。"""
    data = registry if registry is not None else load_registry()
    needle = value.strip().casefold()
    # 步骤1：已经是统一ID时直接使用登记中的规范大小写。
    for entry in data['entries']:
        if entry['id'].casefold() == needle:
            return entry['id']
    # 步骤2：通用别名与作用域内别名分开匹配，不能把不同实验的E1混为一谈。
    matches = set()
    for entry in data['entries']:
        for alias in entry['aliases']:
            if alias['label'].casefold() == needle and (alias['scope'] == 'global' or alias['scope'] == scope):
                matches.add(entry['id'])
    if len(matches) != 1:
        raise ValueError('Unknown/ambiguous method alias; supply canonical ID or explicit scope: ' + value)
    return matches.pop()
