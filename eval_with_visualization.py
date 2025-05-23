import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import numpy as np

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog

# 导入MaskDINO相关配置
from maskdino import add_maskdino_config


def setup_cfg(args):
    # 加载配置文件
    cfg = get_cfg()
    add_maskdino_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # 设置模型权重和置信度阈值
    cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="MaskDINO evaluation with visualization")
    parser.add_argument(
        "--config-file",
        default="configs/custom/custom_maskdino.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--input", help="输入图片目录或单张图片路径")
    parser.add_argument(
        "--output",
        default="./eval_visualizations",
        help="可视化结果保存目录",
    )
    parser.add_argument(
        "--weights",
        default="output/model_final.pth",
        help="模型权重文件路径"
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def main():
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    # 创建预测器
    predictor = DefaultPredictor(cfg)

    # 获取数据集metadata（用于可视化）
    if len(cfg.DATASETS.TEST) > 0:
        metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    else:
        # 如果没有注册数据集，创建一个默认的metadata
        metadata = None

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)

    # 获取输入图片列表
    if os.path.isdir(args.input):
        # 支持常见图片格式
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif"]
        input_files = []
        for ext in image_extensions:
            input_files.extend(glob.glob(os.path.join(args.input, ext)))
            input_files.extend(glob.glob(os.path.join(args.input, ext.upper())))
    else:
        input_files = [args.input]

    print(f"Found {len(input_files)} images for evaluation")

    # 处理每张图片
    for i, path in enumerate(tqdm.tqdm(input_files, desc="Processing images")):
        # 读取图片
        img = read_image(path, format="BGR")

        start_time = time.time()
        # 进行推理
        predictions = predictor(img)
        inference_time = time.time() - start_time

        # 创建可视化器
        v = Visualizer(
            img[:, :, ::-1],  # BGR转RGB
            metadata=metadata,
            scale=0.8,
            instance_mode=ColorMode.IMAGE_BW  # 可以改为IMAGE来保持原始图片颜色
        )

        # 绘制预测结果
        if "instances" in predictions:
            instances = predictions["instances"].to("cpu")
            vis = v.draw_instance_predictions(instances)
        elif "sem_seg" in predictions:
            vis = v.draw_sem_seg(predictions["sem_seg"].argmax(dim=0).to("cpu"))
        elif "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis = v.draw_panoptic_seg_predictions(
                panoptic_seg.to("cpu"), segments_info
            )
        else:
            logger.warning("No valid predictions found for visualization")
            continue

        # 保存可视化结果
        out_filename = os.path.basename(path)
        name, ext = os.path.splitext(out_filename)
        out_filename = f"{name}_vis{ext}"
        out_path = os.path.join(args.output, out_filename)

        # 保存图片 (RGB转BGR用于OpenCV保存)
        cv2.imwrite(out_path, vis.get_image()[:, :, ::-1])

        if i % 10 == 0:
            logger.info(f"Processed {i + 1}/{len(input_files)} images. "
                        f"Inference time: {inference_time:.3f}s. "
                        f"Saved to: {out_path}")

    print(f"\n✅ 所有可视化结果已保存到: {args.output}")
    print(f"总共处理了 {len(input_files)} 张图片")


if __name__ == "__main__":
    main()