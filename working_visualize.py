#!/usr/bin/env python3
"""
基于你已经能运行的train_custom.py的可视化脚本
直接复制你的工作环境设置
"""

import os
import sys
import cv2
import torch
import subprocess
from pathlib import Path
from tqdm import tqdm


def run_eval_and_get_model():
    """
    运行你已经能工作的评估命令，并获取加载的模型
    """
    print("�� 正在加载已验证的模型配置...")

    # 添加项目路径
    sys.path.insert(0, '.')

    try:
        # 导入你项目中的模块（确保使用相同的环境）
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor
        from detectron2.utils.visualizer import Visualizer, ColorMode
        from detectron2.data import MetadataCatalog, DatasetCatalog
        from maskdino import add_maskdino_config

        print("✅ 模块导入成功")
        return True, {
            'get_cfg': get_cfg,
            'DefaultPredictor': DefaultPredictor,
            'Visualizer': Visualizer,
            'ColorMode': ColorMode,
            'MetadataCatalog': MetadataCatalog,
            'DatasetCatalog': DatasetCatalog,
            'add_maskdino_config': add_maskdino_config
        }
    except Exception as e:
        print(f"❌ 模块导入失败: {e}")
        return False, {}


def setup_working_config(modules):
    """
    使用与你的train_custom.py完全相同的配置设置方法
    """
    cfg = modules['get_cfg']()
    modules['add_maskdino_config'](cfg)

    # 不直接加载配置文件，而是手动设置关键参数
    # 基于detectron2的默认配置，只设置必要的参数

    # 模型权重
    cfg.MODEL.WEIGHTS = "output/model_final.pth"

    # 数据集（使用你在配置中定义的）
    cfg.DATASETS.TEST = ("custom_test",)

    # 基本模型配置（避免有问题的参数）
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5

    # 设备配置
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print("✅ 配置设置完成")
    return cfg


def visualize_with_working_setup(output_dir="./working_vis_results", max_images=50):
    """
    使用完全工作的设置进行可视化
    """
    print("�� 使用已验证的配置进行可视化")
    print("=" * 60)

    # 获取模块
    success, modules = run_eval_and_get_model()
    if not success:
        return False

    # 设置配置
    cfg = setup_working_config(modules)

    # 获取数据集
    try:
        dataset_name = cfg.DATASETS.TEST[0]
        dataset_dicts = modules['DatasetCatalog'].get(dataset_name)
        metadata = modules['MetadataCatalog'].get(dataset_name)

        print(f"�� 数据集信息:")
        print(f"   - 名称: {dataset_name}")
        print(f"   - 图像数量: {len(dataset_dicts)}")

    except Exception as e:
        print(f"❌ 获取数据集失败: {e}")
        print("�� 尝试查找可用的数据集...")

        # 列出所有可用的数据集
        from detectron2.data.catalog import DatasetCatalog
        available_datasets = list(DatasetCatalog.list())
        print(f"可用数据集: {available_datasets}")

        # 尝试使用包含'custom'或'test'的数据集
        test_datasets = [d for d in available_datasets if 'test' in d.lower() or 'custom' in d.lower()]
        if test_datasets:
            dataset_name = test_datasets[0]
            print(f"使用数据集: {dataset_name}")
            dataset_dicts = modules['DatasetCatalog'].get(dataset_name)
            metadata = modules['MetadataCatalog'].get(dataset_name)
        else:
            print("❌ 没有找到合适的测试数据集")
            return False

    # 创建预测器
    try:
        predictor = modules['DefaultPredictor'](cfg)
        print("✅ 预测器创建成功")
    except Exception as e:
        print(f"❌ 预测器创建失败: {e}")
        return False

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 开始可视化
    print(f"�� 开始生成可视化结果...")
    success_count = 0

    for i, d in enumerate(tqdm(dataset_dicts[:max_images], desc="生成可视化")):
        try:
            # 读取图像
            im = cv2.imread(d["file_name"])
            if im is None:
                continue

            # 预测
            outputs = predictor(im)

            # 检查预测结果
            if "instances" not in outputs:
                continue

            instances = outputs["instances"].to("cpu")
            if len(instances) == 0:
                continue

            # 过滤低置信度预测
            if hasattr(instances, 'scores'):
                high_conf = instances.scores > 0.5
                if high_conf.sum() == 0:
                    continue
                instances = instances[high_conf]

            # 可视化
            v = modules['Visualizer'](
                im[:, :, ::-1],
                metadata=metadata,
                scale=1.0,
                instance_mode=modules['ColorMode'].IMAGE_BW
            )

            out = v.draw_instance_predictions(instances)

            # 保存
            base_name = Path(d["file_name"]).stem
            filename = f"working_vis_{i:04d}_{base_name}.jpg"
            output_path = os.path.join(output_dir, filename)
            out.save(output_path)

            success_count += 1

        except Exception as e:
            print(f"⚠️  图像 {i} 处理失败: {e}")
            continue

    print(f"\n✅ 可视化完成!")
    print(f"   - 成功处理: {success_count} 张图像")
    print(f"   - 输出目录: {output_dir}")

    return success_count > 0


def main():
    import argparse

    parser = argparse.ArgumentParser(description="基于工作配置的可视化脚本")
    parser.add_argument("--output-dir", default="./working_vis_results")
    parser.add_argument("--max-images", type=int, default=50)

    args = parser.parse_args()

    # 检查权重文件
    if not os.path.exists("output/model_final.pth"):
        print("❌ 权重文件不存在: output/model_final.pth")
        return

    success = visualize_with_working_setup(args.output_dir, args.max_images)

    if success:
        print("�� 可视化成功完成!")
    else:
        print("❌ 可视化失败")


if __name__ == "__main__":
    main()