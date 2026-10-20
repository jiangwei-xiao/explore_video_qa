"""固定视频包的CPU预提取：独立写入，完成后以不可覆盖凭证供原资格流程复用。"""
import json
import os
from pathlib import Path
import time

from videoqa_set_value import longvideo_assets as assets


def preextract(asset_root, state_root, needed_ids):
    """接口：仅预提取已完整验收的固定包，不抽样、不读答案、不调用模型。

    原资格协调器可能同时进入提取，因此使用独立成员目录及硬链接原子发布：
    不覆盖协调器已经生成的凭证，也不共享其成员暂存文件。
    """
    asset_root=Path(asset_root);state_root=Path(state_root)
    with assets.lock(state_root/'preextract.lock'):
        records=[]
        # 步骤1：固定总体仍为原八包；只提前处理当前完整包，缺包留给原协调器。
        for entry in assets.inventory():
            if entry['kind']!='archive':continue
            source_receipt=asset_root/'verified'/(entry['name']+'.json')
            if not source_receipt.exists():continue
            canonical=asset_root/'extraction'/(entry['name']+'.json')
            if canonical.exists():continue
            stage=state_root/'packages'/entry['name']
            receipt=stage/'verified'/(entry['name']+'.json')
            receipt.parent.mkdir(parents=True,exist_ok=True)
            if not receipt.exists():os.link(source_receipt,receipt)
            started=time.time()
            assets.atomic_json(state_root/'heartbeat.json',dict(stage='extracting_verified_archive',
                archive=entry['name'],pid=os.getpid(),sid=os.getsid(0),unix=started,model_calls=0))
            # 步骤2：调用冻结安全提取器，重新核验整包SHA、合法成员与完整EOF。
            result=assets.extract_archive(stage,entry['name'],needed_ids)
            staged=stage/'extraction'/(entry['name']+'.json')
            if result['archive_sha256']!=entry['sha256']:raise ValueError('Archive identity mismatch')
            canonical.parent.mkdir(parents=True,exist_ok=True)
            # 步骤3：link是不可覆盖的原子发布；若原流程先完成，则保留其原凭证。
            try:
                os.link(staged,canonical);published=True
            except FileExistsError:
                existing=json.loads(canonical.read_text())
                expected={(r['video_id'],r['sha256'],r['bytes']) for r in result['members']}
                observed={(r['video_id'],r['sha256'],r['bytes']) for r in existing['members']}
                if existing['archive_sha256']!=entry['sha256'] or observed!=expected:
                    raise ValueError('Concurrent extraction identity mismatch')
                published=False
            records.append(dict(archive=entry['name'],videos=len(result['members']),
                published=published,seconds=time.time()-started,receipt=str(canonical)))
            assets.atomic_json(state_root/'progress.json',dict(records=records,model_calls=0,unix=time.time()))
        # 步骤4：这里只结束预提取，不发布资格总体或宣称1200视频已满足条件。
        result=dict(status='completed_available_archives',records=records,model_calls=0,unix=time.time())
        assets.atomic_json(state_root/'completed.json',result)
        return result
