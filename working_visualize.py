#!/usr/bin/env python3
"""
基于文件夹的可视化脚本 - 不依赖数据集注册
直接从图像文件夹读取图像进行预测和可视化
"""

import os
import cv2
import torch
import argparse
import sys
from pathlib import Path
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, '.')


def setup_model_and_config(weights_path, confidence_threshold=0.5):
    """
    设置模型和配置，不依赖外部配置文件
    """
    print("�� 设置模型配置...")

    try:
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor
        from detectron2.utils.visualizer import Visualizer, ColorMode
        from detectron2.data import MetadataCatalog
        from maskdino import add_maskdino_config

        # 创建配置
        cfg = get_cfg()
        add_maskdino_config(cfg)

        # 手动设置基本配置，避免复杂的配置文件
        cfg.MODEL.WEIGHTS = weights_path
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        # 基本模型设置
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5

        # MaskDINO 特定设置（使用默认值）
        cfg.MODEL.MaskDINO.HIDDEN_DIM = 256
        cfg.MODEL.MaskDINO.NUM_OBJECT_QUERIES = 100
        cfg.MODEL.MaskDINO.NHEADS = 8
        cfg.MODEL.MaskDINO.DROPOUT = 0.0
        cfg.MODEL.MaskDINO.DIM_FEEDFORWARD = 2048
        cfg.MODEL.MaskDINO.ENC_LAYERS = 6
        cfg.MODEL.MaskDINO.DEC_LAYERS = 6
        cfg.MODEL.MaskDINO.PRE_NORM = False

        # 输入设置
        cfg.INPUT.MIN_SIZE_TEST = 800
        cfg.INPUT.MAX_SIZE_TEST = 1333

        cfg.freeze()

        # 创建预测器
        predictor = DefaultPredictor(cfg)

        # 创建简单的元数据（用于可视化）
        # 如果没有注册的数据集，我们创建一个基本的元数据
        metadata = type('SimpleMetadata', (), {
            'thing_classes': [f'class_{i}' for i in range(80)],  # COCO的80个类别
            'thing_colors': None
        })()

        print("✅ 模型设置完成")
        return predictor, Visualizer, ColorMode, metadata

    except Exception as e:
        print(f"❌ 模型设置失败: {e}")
        return None, None, None, None


def find_images_in_folder(folder_path, extensions=['.jpg', '.jpeg', '.png', '.bmp']):
    """
    在文件夹中查找图像文件
    """
    folder = Path(folder_path)
    if not folder.exists():
        print(f"❌ 文件夹不存在: {folder_path}")
        return []

    image_files = []
    for ext in extensions:
        image_files.extend(folder.rglob(f'*{ext}'))
        image_files.extend(folder.rglob(f'*{ext.upper()}'))

    print(f"��️  在 {folder_path} 中找到 {len(image_files)} 张图像")
    return sorted(image_files)


def auto_find_image_folders():
    """
    自动查找可能包含图像的文件夹
    """
    print("�� 自动查找图像文件夹...")

    possible_folders = [
        'datasets',
        'data',
        'images',
        'test_images',
        'val_images',
        'custom_images',
        './datasets',
        './data',
        '../datasets',
        '../data',
    ]

    found_folders = []
    for folder in possible_folders:
        folder_path = Path(folder)
        if folder_path.exists():
            images = find_images_in_folder(folder_path)
            if images:
                found_folders.append((folder_path, len(images)))

    if found_folders:
        print("�� 找到的图像文件夹:")
        for folder, count in found_folders:
            print(f"   - {folder}: {count} 张图像")

        # 返回图像最多的文件夹
        best_folder = max(found_folders, key=lambda x: x[1])[0]
        print(f"�� 推荐使用: {best_folder}")
        return str(best_folder)
    else:
        print("❌ 没有找到包含图像的文件夹")
        return None


def visualize_images_from_folder(image_folder, weights_path, output_dir,
                                 max_images=50, confidence_threshold=0.5):
    """
    从文件夹中读取图像并进行可视化
    """
    print("�� 开始基于文件夹的可视化")
    print("=" * 60)

    # 设置模型
    predictor, Visualizer, ColorMode, metadata = setup_model_and_config(
        weights_path, confidence_threshold
    )

    if predictor is None:
        return False

    # 查找图像
    if image_folder is None:
        image_folder = auto_find_image_folders()
        if image_folder is None:
            return False

    image_files = find_images_in_folder(image_folder)
    if not image_files:
        return False

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 处理图像
    print(f"�� 开始处理 {min(max_images, len(image_files))} 张图像...")

    success_count = 0
    error_count = 0

    for i, img_path in enumerate(tqdm(image_files[:max_images], desc="处理图像")):
        try:
            # 读取图像
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"⚠️  无法读取: {img_path}")
                error_count += 1
                continue

            # 预测
            outputs = predictor(image)

            # 检查预测结果
            if "instances" not in outputs:
                continue

            instances = outputs["instances"].to("cpu")
            if len(instances) == 0:
                continue

            # 过滤低置信度预测
            if hasattr(instances, 'scores'):
                high_conf = instances.scores > confidence_threshold
                if high_conf.sum() == 0:
                    continue
                instances = instances[high_conf]

            # 可视化
            v = Visualizer(
                image[:, :, ::-1],  # BGR to RGB
                metadata=metadata,
                scale=1.0,
                instance_mode=ColorMode.IMAGE_BW
            )

            vis_output = v.draw_instance_predictions(instances)

            # 保存
            output_filename = f"folder_vis_{i:04d}_{img_path.stem}.jpg"
            output_path = os.path.join(output_dir, output_filename)
            vis_output.save(output_path)

            success_count += 1

            # 显示部分预测信息
            if hasattr(instances, 'scores') and len(instances.scores) > 0:
                max_score = instances.scores.max().item()
                num_detections = len(instances)
                if i < 5:  # 只显示前5张图像的详细信息
                    print(f"   图像 {i + 1}: {num_detections} 个检测, 最高置信度: {max_score:.3f}")

        except Exception as e:
            print(f"⚠️  处理失败 {img_path.name}: {e}")
            error_count += 1
            continue

    print(f"\n�� 处理完成:")
    print(f"   ✅ 成功: {success_count}")
    print(f"   ❌ 失败: {error_count}")
    print(f"   �� 输出目录: {output_dir}")

    if success_count > 0:
        print(f"\n�� 可视化成功! 查看结果:")
        print(f"   ls {output_dir}/")

        # 显示第一个结果文件
        first_result = Path(output_dir).glob("folder_vis_*.jpg")
        try:
            first_file = next(first_result)
            print(f"   第一个结果: {first_file}")
        except StopIteration:
            pass

    return success_count > 0


def main():
    parser = argparse.ArgumentParser(description="基于文件夹的MaskDINO可视化")
    parser.add_argument(
        "--image-folder",
        help="包含图像的文件夹路径（如果不指定，会自动查找）"
    )
    parser.add_argument(
        "--weights",
        default="output/model_final.pth",
        help="模型权重文件路径"
    )
    parser.add_argument(
        "--output-dir",
        default="./folder_vis_results",
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

    # 检查权重文件
    if not os.path.exists(args.weights):
        print(f"❌ 权重文件不存在: {args.weights}")
        return

    print(f"�� 使用模型: {args.weights}")
    print(f"�� 置信度阈值: {args.confidence_threshold}")
    print(f"�� 最大图像数: {args.max_images}")

    # 运行可视化
    success = visualize_images_from_folder(
        args.image_folder,
        args.weights,
        args.output_dir,
        args.max_images,
        args.confidence_threshold
    )

    if success:
        print("\n�� 可视化完成!")
    else:
        print("\n❌ 可视化失败!")
        print("\n�� 故障排除建议:")
        print("1. 检查图像文件夹是否存在且包含图像")
        print("2. 检查模型权重文件是否正确")
        print("3. 尝试降低置信度阈值 --confidence-threshold 0.3")


if __name__ == "__main__":
    main()