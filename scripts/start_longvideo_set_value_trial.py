"""独立后台启动器：显式解释器还须配套工具PATH，不依赖交互式Conda激活。"""
import argparse
import os
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
PREP_BIN=Path('/App/conda/envs/explore_video_qa_lmms_qwen/bin')


def launch(run_id):
    """接口：启动原冻结协调器；只修复进程环境，不改变阶段、算法或调用预算。

    本函数不终止已有协调器；原入口排他锁继续阻止重复执行。
    """
    if not run_id.replace('_','').isalnum():raise ValueError('Invalid isolated run ID')
    # 步骤1：固定工具目录先于系统PATH，保证非交互式子进程能找到本环境ffprobe。
    env=dict(os.environ)
    env['PATH']=str(PREP_BIN)+os.pathsep+env.get('PATH',os.defpath)
    for name in ('python','ffprobe','ffmpeg'):
        if shutil.which(name,path=env['PATH'])!=str(PREP_BIN/name):
            raise RuntimeError('Missing pinned environment executable: '+name)
    # 步骤2：exec保持后台PID/SID，后续验收和恢复全部由原协调器及其账本负责。
    python=str(PREP_BIN/'python')
    os.chdir(ROOT)
    os.execve(python,[python,'-u',str(ROOT/'scripts/run_longvideo_set_value_trial.py'),
        '--stage','run','--run-id',run_id],env)


def main():
    """接口：只接收原运行编号；恢复不创建第二套实验或重新抽样。"""
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',default='set_value_longvideo__llava8__train800_val400__20261019_r1')
    launch(parser.parse_args().run_id)


if __name__=='__main__':main()
