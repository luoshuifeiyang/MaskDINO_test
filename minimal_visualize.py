#!/usr/bin/env python3
"""
最小化可视化脚本 - 避免复杂的配置问题
这个脚本直接使用train_custom.py中已经验证过的配置方式
"""

import os
import cv2
import torch
import argparse
import sys
from pathlib import Path
from tqdm import tqdm

# 添加项目根目录到Python路径
sys.path.insert(0, '.')


def setup_environment():
    """设置环境和导入"""
    try:
        # 基本detectron2导入
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor
        from detectron2.utils.visualizer import Visualizer, ColorMode
        from detectron2.data import MetadataCatalog, DatasetCatalog
        from detectron2.utils.logger import setup_logger

        # MaskDINO导入
        from maskdino import add_maskdino_config

        return True, {
            'get_cfg': get_cfg,
            'DefaultPredictor': DefaultPredictor,
            'Visualizer': Visualizer,
            'ColorMode': ColorMode,
            'MetadataCatalog': MetadataCatalog,
            'DatasetCatalog': DatasetCatalog,
            'setup_logger': setup_logger,
            'add_maskdino_config': add_maskdino_config
        }
    except ImportError as e:
        print(f"导入错误: {e}")
        return False, {}


def minimal_setup_cfg(config_file, weights_path, modules):
    """
    最小化配置设置，复制train_custom.py的逻辑
    """
    cfg = modules['get_cfg']()

    # 添加MaskDINO配置
    modules['add_maskdino_config'](cfg)

    # 尝试直接执行eval命令中使用的设置
    try:
        # 先尝试直接加载
        cfg.merge_from_file(config_file)
        print("✅ 配置文件加载成功")
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        return None

    # 设置模型权重和评估模式
    cfg.MODEL.WEIGHTS = weights_path
    cfg.freeze()

    return cfg


def get_dataset_from_config(cfg, modules):
    """
    从配置中获取数据集信息
    """
    try:
        # 获取测试数据集名称
        if len(cfg.DATASETS.TEST) > 0:
            dataset_name = cfg.DATASETS.TEST[0]
            print(f"使用数据集: {dataset_name}")

            # 获取数据集字典和元数据
            dataset_dicts = modules['DatasetCatalog'].get(dataset_name)
            metadata = modules['MetadataCatalog'].get(dataset_name)

            return dataset_dicts, metadata, dataset_name
        else:
            print("❌ 配置中没有测试数据集")
            return None, None, None
    except Exception as e:
        print(f"❌ 获取数据集失败: {e}")
        return None, None, None


def run_minimal_visualization(config_file, weights_path, output_dir, max_images=50, confidence_threshold=0.5):
    """
    运行最小化可视化
    """
    print("�� 启动最小化可视化脚本")
    print("=" * 50)

    # 设置环境
    success, modules = setup_environment()
    if not success:
        print("❌ 环境设置失败")
        return False

    # 设置日志
    modules['setup_logger']()

    # 设置配置
    print("�� 加载配置...")
    cfg = minimal_setup_cfg(config_file, weights_path, modules)
    if cfg is None:
        return False

    # 获取数据集
    print("�� 获取数据集...")
    dataset_dicts, metadata, dataset_name = get_dataset_from_config(cfg, modules)
    if dataset_dicts is None:
        return False

    print(f"   - 数据集名称: {dataset_name}")
    print(f"   - 图像总数: {len(dataset_dicts)}")
    print(f"   - 处理数量: {min(max_images, len(dataset_dicts))}")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 创建预测器
    print("�� 创建预测器...")
    try:
        predictor = modules['DefaultPredictor'](cfg)
        print("✅ 预测器创建成功")
    except Exception as e:
        print(f"❌ 预测器创建失败: {e}")
        return False

    # 处理图像
    print("�� 开始生成可视化...")
    success_count = 0
    error_count = 0

    for i, d in enumerate(tqdm(dataset_dicts[:max_images], desc="处理图像")):
        try:
            # 读取图像
            im = cv2.imread(d["file_name"])
            if im is None:
                print(f"⚠️  无法读取图像: {d['file_name']}")
                error_count += 1
                continue

            # 预测
            outputs = predictor(im)

            # 过滤低置信度预测
            if "instances" in outputs:
                instances = outputs["instances"].to("cpu")
                if hasattr(instances, 'scores'):
                    # 只保留高置信度的预测
                    high_conf_mask = instances.scores > confidence_threshold
                    if high_conf_mask.sum() == 0:
                        continue  # 没有高置信度预测，跳过
                    instances = instances[high_conf_mask]
            else:
                continue

            # 可视化
            v = modules['Visualizer'](
                im[:, :, ::-1],
                metadata=metadata,
                scale=1.0,
                instance_mode=modules['ColorMode'].IMAGE_BW
            )

            out = v.draw_instance_predictions(instances)

            # 保存
            image_id = d.get("image_id", i)
            base_name = Path(d["file_name"]).stem
            filename = f"vis_{i:04d}_{base_name}_id_{image_id}.jpg"
            output_path = os.path.join(output_dir, filename)

            out.save(output_path)
            success_count += 1

        except Exception as e:
            print(f"⚠️  处理图像失败 {i}: {e}")
            error_count += 1
            continue

    print("\n" + "=" * 50)
    print("�� 处理完成统计:")
    print(f"   ✅ 成功: {success_count}")
    print(f"   ❌ 失败: {error_count}")
    print(f"   �� 输出目录: {output_dir}")

    return success_count > 0


def main():
    parser = argparse.ArgumentParser(description="MaskDINO 最小化可视化脚本")
    parser.add_argument(
        "--config-file",
        default="configs/custom/custom_maskdino.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--weights",
        default="output/model_final.pth",
        help="模型权重路径"
    )
    parser.add_argument(
        "--output-dir",
        default="./minimal_vis_results",
        help="输出目录"
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=50,
        help="最大处理图像数量"
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="置信度阈值"
    )

    args = parser.parse_args()

    # 检查文件是否存在
    if not os.path.exists(args.config_file):
        print(f"❌ 配置文件不存在: {args.config_file}")
        return

    if not os.path.exists(args.weights):
        print(f"❌ 权重文件不存在: {args.weights}")
        return

    # 运行可视化
    success = run_minimal_visualization(
        args.config_file,
        args.weights,
        args.output_dir,
        args.max_images,
        args.confidence_threshold
    )

    if success:
        print("�� 可视化完成！")
    else:
        print("❌ 可视化失败！")


if __name__ == "__main__":
    main()