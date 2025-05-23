#!/usr/bin/env python3
"""
可视化验证脚本 - 基于MaskDINO
生成验证集的预测结果可视化图像并保存到本地
"""

import os
import cv2
import torch
import logging
import argparse
from pathlib import Path
from tqdm import tqdm

# detectron2 imports
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader, DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.logger import setup_logger

# MaskDINO specific imports
from maskdino import add_maskdino_config


def setup_cfg(args):
    """
    创建配置并执行基本设置
    """
    cfg = get_cfg()
    # 添加MaskDINO特定配置
    add_maskdino_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # 设置模型权重
    cfg.MODEL.WEIGHTS = args.weights

    # 设置为评估模式
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold

    cfg.freeze()
    return cfg


def visualize_predictions(cfg, output_dir, max_images=None):
    """
    对验证集进行预测并可视化结果

    Args:
        cfg: detectron2配置
        output_dir: 输出目录
        max_images: 最大处理图像数量（None表示处理所有）
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 创建预测器
    predictor = DefaultPredictor(cfg)

    # 获取数据集名称
    dataset_name = cfg.DATASETS.TEST[0]

    # 获取数据加载器
    data_loader = build_detection_test_loader(cfg, dataset_name)

    # 获取数据集元数据
    metadata = MetadataCatalog.get(dataset_name)

    logger = logging.getLogger(__name__)
    logger.info(f"开始可视化验证，输出目录: {output_dir}")
    logger.info(f"数据集: {dataset_name}")
    logger.info(f"类别数: {len(metadata.thing_classes) if hasattr(metadata, 'thing_classes') else '未知'}")

    processed_count = 0

    # 遍历验证集
    for idx, inputs in enumerate(tqdm(data_loader, desc="生成可视化结果")):
        if max_images and processed_count >= max_images:
            break

        for input_per_image in inputs:
            if max_images and processed_count >= max_images:
                break

            # 读取图像
            file_name = input_per_image["file_name"]
            image = cv2.imread(file_name)

            if image is None:
                logger.warning(f"无法读取图像: {file_name}")
                continue

            # 进行预测
            predictions = predictor(image)

            # 创建可视化器
            v = Visualizer(
                image[:, :, ::-1],  # BGR -> RGB
                metadata=metadata,
                scale=1.0,
                instance_mode=ColorMode.IMAGE_BW  # 可选：IMAGE, IMAGE_BW, SEGMENTATION
            )

            # 绘制预测结果
            if "instances" in predictions:
                instances = predictions["instances"].to("cpu")
                vis_output = v.draw_instance_predictions(instances)
            elif "panoptic_seg" in predictions:
                panoptic_seg, segments_info = predictions["panoptic_seg"]
                vis_output = v.draw_panoptic_seg_predictions(
                    panoptic_seg.to("cpu"), segments_info
                )
            else:
                logger.warning(f"未找到有效预测结果: {file_name}")
                continue

            # 保存结果
            image_id = input_per_image.get("image_id", processed_count)
            base_name = Path(file_name).stem
            output_filename = f"{base_name}_pred_{image_id}.jpg"
            output_path = os.path.join(output_dir, output_filename)

            # 保存可视化结果
            vis_output.save(output_path)

            # 可选：同时保存原始预测数据
            if args.save_predictions:
                pred_data = {
                    "file_name": file_name,
                    "image_id": image_id,
                    "predictions": predictions
                }
                pred_filename = f"{base_name}_pred_data_{image_id}.pth"
                pred_path = os.path.join(output_dir, "predictions", pred_filename)
                os.makedirs(os.path.dirname(pred_path), exist_ok=True)
                torch.save(pred_data, pred_path)

            processed_count += 1

            if processed_count % 50 == 0:
                logger.info(f"已处理 {processed_count} 张图像")

    logger.info(f"可视化完成! 共处理 {processed_count} 张图像")
    logger.info(f"结果保存在: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="MaskDINO验证可视化")
    parser.add_argument(
        "--config-file",
        default="configs/custom/custom_maskdino.yaml",
        metavar="FILE",
        help="配置文件路径"
    )
    parser.add_argument(
        "--weights",
        default="output/model_final.pth",
        help="模型权重文件路径"
    )
    parser.add_argument(
        "--output-dir",
        default="./visualization_results",
        help="可视化结果输出目录"
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="置信度阈值"
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="最大处理图像数量（用于快速测试）"
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="是否保存原始预测数据"
    )
    parser.add_argument(
        "opts",
        help="修改配置选项",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    # 设置日志
    setup_logger()

    # 设置配置
    cfg = setup_cfg(args)

    # 运行可视化
    visualize_predictions(cfg, args.output_dir, args.max_images)


if __name__ == "__main__":
    # 导入ColorMode（需要在main中导入以避免循环导入）
    from detectron2.utils.visualizer import ColorMode

    main()