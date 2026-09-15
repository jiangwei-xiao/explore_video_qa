"""证据审计命令入口：不调用实验模型，中文接口详见独立审计模块。"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from videoqa_audit.core import main

if __name__ == '__main__':
    main()
