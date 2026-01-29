#!/usr/bin/env python
import os
import sys

# 设置Hugging Face镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 修改sys.path
sys.path.insert(0, '/root/CRA-GQA-main')

# 导入主模块
import main

# 获取参数（假设main模块有parse_args函数）
args = main.parse_args() if hasattr(main, 'parse_args') else None

# 获取配置（假设有parse_yaml函数）
if hasattr(main, 'parse_yaml'):
    cfgs = main.parse_yaml(args.config) if args and hasattr(args, 'config') else {}
    
    # 强制修改配置中的模型路径
    if 'model' in cfgs and 'lan_weight_path' in cfgs['model']:
        cfgs['model']['lan_weight_path'] = "roberta-base"
        print("已修改配置中的模型路径为: roberta-base")
    
    # 运行主函数
    main.main(args)
else:
    # 直接运行main函数
    main.main(args) if hasattr(main, 'main') else None
