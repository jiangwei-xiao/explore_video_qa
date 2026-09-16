"""将新增三题接入现有103-1静态服务目录，不改原实验结果。"""
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from videoqa_runtime.common import read_json,write_json,sha256
from html.parser import HTMLParser


class Links(HTMLParser):
    """提取本地链接以核对浏览入口，避免仅检查文件存在。"""
    def __init__(self):
        """初始化链接容器。"""
        super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        """只收集资源地址，不执行网页内容。"""
        self.links.extend(v.split('#')[0] for k,v in attrs if k in ('src','href'))


def publish(source,walkthrough):
    """接口：验证完整生成清单后创建独立子目录，并只增加旧入口的导航链接。"""
    source=Path(source);walkthrough=Path(walkthrough)
    # 步骤1：所有产物及HTML引用必须闭合，未完成输出禁止发布。
    manifest=read_json(source/'artifact_manifest.json')
    assert all(sha256(source/p)==h for p,h in manifest.items())
    for p in source.rglob('*.html'):
        links=Links();links.feed(p.read_text());assert all((p.parent/u).exists() for u in links.links)
    # 步骤2：新增目录，不覆盖已有案例或用户记录。
    target=walkthrough/'additional_three';shutil.copytree(source,target)
    index=walkthrough/'index.html';page=index.read_text()
    link='<p><a href="additional_three/index.html">新增三题：395-2、383-2、544-1逐阶段review</a></p>'
    assert link not in page
    page=page.replace('<h1>103-1：从原片到最终16帧</h1>','<h1>103-1：从原片到最终16帧</h1>'+link)
    index.write_text(page)
    write_json(walkthrough/'artifact_manifest.json',{str(p.relative_to(walkthrough)):sha256(p) for p in walkthrough.rglob('*') if p.is_file() and p!=walkthrough/'artifact_manifest.json'})
    print('published',len(manifest),'verified files')


if __name__=='__main__':publish(sys.argv[1],sys.argv[2])
