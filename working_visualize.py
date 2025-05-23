#!/usr/bin/env python3
"""
MaskDINO 可视化脚本
"""
import argparse
import os
import cv2
import torch
import numpy as np
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.projects.deeplab import add_deeplab_config
from tqdm import tqdm
import glob

# 添加MaskDINO配置
import sys

sys.path.append(".")
from maskdino import add_maskdino_config

# 导入train_custom以触发数据集注册
import train_custom


def get_all_images(image_folder):
    """获取文件夹中所有图像文件，包括子文件夹"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.JPG', '.JPEG', '.PNG', '.BMP', '.TIFF',
                        '.TIF'}
    image_files = []

    for root, dirs, files in os.walk(image_folder):
        for file in files:
            if any(file.endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(root, file))

    return sorted(image_files)


def visualize_clean_masks(image, instances, metadata):
    """
    只显示mask，不显示边界框、置信度和类别名称
    """
    # 创建可视化器
    v = Visualizer(image[:, :, ::-1], metadata=metadata, scale=1.0, instance_mode=ColorMode.IMAGE)

    if len(instances) > 0 and instances.has("pred_masks"):
        masks = instances.pred_masks

        # 为每个mask分配随机颜色
        colors = []
        for i in range(len(masks)):
            colors.append(np.random.rand(3))

        # 只绘制masks，不绘制框和标签
        for i, mask in enumerate(masks):
            mask_array = mask.numpy().astype(bool)
            v.draw_binary_mask(mask_array, color=colors[i], alpha=0.6)

    return v.get_output()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default="configs/custom/custom_maskdino.yaml")
    parser.add_argument("--image-folder", required=True)
    parser.add_argument("--output-dir", default="./vis_output")
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--weights", required=True)
    # 移除max-images限制，改为可选参数，默认处理所有图像
    parser.add_argument("--max-images", type=int, default=None, help="最大处理图像数量，不设置则处理所有图像")
    parser.add_argument("--debug", action="store_true", help="显示调试信息")
    # 添加选项来控制是否显示边界框和标签
    parser.add_argument("--clean-output", action="store_true", default=True, help="只显示mask，不显示边界框和标签")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    print(f"配置文件: {args.config_file}")
    print(f"模型权重: {args.weights}")
    print(f"置信度阈值: {args.confidence_threshold}")
    print(f"清洁输出模式: {args.clean_output}")

    # 设置配置
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # 覆盖模型权重
    cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # 设置阈值 - MaskDINO使用不同的配置结构
    # 检查并设置各种可能的阈值配置
    if hasattr(cfg.MODEL, 'RETINANET'):
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    if hasattr(cfg.MODEL, 'ROI_HEADS'):
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    if hasattr(cfg.MODEL, 'PANOPTIC_FPN'):
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold

    # MaskDINO特定的阈值设置
    if hasattr(cfg.MODEL, 'MASKDINO'):
        if hasattr(cfg.MODEL.MASKDINO, 'TEST'):
            cfg.MODEL.MASKDINO.TEST.DETECTION_SCORE_THRESH = args.confidence_threshold
            cfg.MODEL.MASKDINO.TEST.OVERLAP_THRESH = 0.8
            cfg.MODEL.MASKDINO.TEST.OBJECT_MASK_THRESHOLD = 0.8

    cfg.freeze()

    # 创建预测器
    predictor = DefaultPredictor(cfg)

    # 获取元数据
    dataset_name = cfg.DATASETS.TEST[0] if cfg.DATASETS.TEST else "custom_test"
    print(f"使用数据集: {dataset_name}")

    metadata = MetadataCatalog.get(dataset_name)
    print(f"类别: {metadata.thing_classes if hasattr(metadata, 'thing_classes') else 'Unknown'}")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 获取图像列表 - 修改这里来处理所有图像
    print("正在扫描图像文件...")
    image_files = get_all_images(args.image_folder)

    # 如果设置了max_images，则限制数量
    if args.max_images is not None:
        image_files = image_files[:args.max_images]
        print(f"限制处理图像数量为: {args.max_images}")

    print(f"找到 {len(image_files)} 张图像")

    if len(image_files) == 0:
        print("未找到任何图像文件！")
        return

    success = 0
    failed = 0
    no_detection = 0

    # 处理图像
    for idx, img_path in enumerate(tqdm(image_files, desc="处理图像")):
        try:
            # 读取图像
            img = cv2.imread(img_path)
            if img is None:
                print(f"无法读取: {os.path.basename(img_path)}")
                failed += 1
                continue

            # 推理
            outputs = predictor(img)

            # 调试信息
            if args.debug and idx < 3:  # 只显示前3张图的调试信息
                print(f"\n=== 调试信息 ===")
                print(f"图像 #{idx}: {os.path.basename(img_path)}")
                print(f"图像尺寸: {img.shape}")
                print(f"输出键: {list(outputs.keys())}")

                if "instances" in outputs:
                    instances = outputs["instances"]
                    print(f"检测数量: {len(instances)}")
                    if len(instances) > 0:
                        print(f"实例字段: {instances.get_fields().keys()}")
                        if hasattr(instances, 'scores'):
                            scores = instances.scores.cpu().numpy()
                            print(f"得分: {scores[:5]}")
                        if hasattr(instances, 'pred_classes'):
                            classes = instances.pred_classes.cpu().numpy()
                            print(f"类别ID: {classes[:5]}")
                            # 检查类别ID是否超出范围
                            if hasattr(metadata, 'thing_classes'):
                                num_classes = len(metadata.thing_classes)
                                if any(c >= num_classes for c in classes):
                                    print(f"警告: 某些类别ID超出范围 (最大值应为{num_classes - 1})")

            # 处理输出
            vis_output = None
            if "instances" in outputs:
                instances = outputs["instances"].to("cpu")
                if len(instances) > 0:
                    # 过滤掉得分低于阈值的实例
                    if hasattr(instances, 'scores'):
                        keep = instances.scores > args.confidence_threshold
                        instances = instances[keep]

                    if len(instances) > 0:
                        # 根据clean_output参数选择可视化方式
                        if args.clean_output:
                            # 只显示mask，不显示边界框和标签
                            vis_output = visualize_clean_masks(img, instances, metadata)
                        else:
                            # 显示完整的检测结果（边界框、标签、置信度）
                            v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.0)
                            vis_output = v.draw_instance_predictions(instances)
                    else:
                        # 过滤后没有实例
                        v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.0)
                        vis_output = v.output
                        no_detection += 1
                else:
                    # 没有检测到物体
                    v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.0)
                    vis_output = v.output
                    no_detection += 1
            else:
                print(f"未知输出格式: {list(outputs.keys())}")
                failed += 1
                continue

            # 保存结果
            out_path = os.path.join(args.output_dir, os.path.basename(img_path))
            vis_img = vis_output.get_image()[:, :, ::-1]
            cv2.imwrite(out_path, vis_img)
            success += 1

        except Exception as e:
            print(f"\n处理失败 {os.path.basename(img_path)}: {str(e)}")
            if args.debug:
                import traceback
                traceback.print_exc()
            failed += 1

    print(f"\n完成!")
    print(f"  成功: {success}")
    print(f"  失败: {failed}")
    print(f"  无检测结果: {no_detection}")
    print(f"  输出目录: {args.output_dir}")


if __name__ == "__main__":
    main()