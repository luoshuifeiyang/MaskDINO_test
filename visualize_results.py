# visualize_results.py
import os
import cv2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import glob

# 配置
cfg = get_cfg()
cfg.merge_from_file("configs/custom/custom_maskdino.yaml")
cfg.MODEL.WEIGHTS = "output/model_final.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # 设置阈值

# 创建预测器
predictor = DefaultPredictor(cfg)

# 创建输出目录
output_dir = "output/visualizations"
os.makedirs(output_dir, exist_ok=True)

# 获取测试图像
test_images = glob.glob("/media/ubuntu/DATA4T/dataset/LIS_paixu/RGB-dark/images/test/*.jpg")  # 修改为你的测试图像路径

for idx, image_path in enumerate(test_images[:50]):  # 处理前50张
    img = cv2.imread(image_path)
    outputs = predictor(img)

    # 可视化
    v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TEST[0]))
    vis = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    # 保存
    output_path = os.path.join(output_dir, f"result_{idx:04d}.png")
    cv2.imwrite(output_path, vis.get_image()[:, :, ::-1])
    print(f"Saved: {output_path}")