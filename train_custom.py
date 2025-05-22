#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.abspath('.'))

# 导入数据集注册
from register_dataset import register_custom_dataset

# 注册数据集
register_custom_dataset()

# 导入原始训练脚本的所有功能
from train_net import *

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )