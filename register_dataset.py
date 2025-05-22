
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog
import os


def register_custom_dataset():
    # 数据集根目录 - 修改为你的实际路径
    DATASET_ROOT = "/media/ubuntu/DATA4T/dataset/LIS_paixu/RGB-dark"

    # 检查路径是否存在
    if not os.path.exists(DATASET_ROOT):
        raise FileNotFoundError(f"数据集路径不存在: {DATASET_ROOT}")

    # 注册训练集
    train_json = os.path.join(DATASET_ROOT, "annotations/lis_coco_JPG_train+1.json")
    train_images = os.path.join(DATASET_ROOT, "images")



    # 注册验证集
    val_json = os.path.join(DATASET_ROOT, "annotations/lis_coco_JPG_test+1.json")
    val_images = os.path.join(DATASET_ROOT, "images")



    # 注册测试集
    test_json = os.path.join(DATASET_ROOT, "annotations/lis_coco_JPG_test+1.json")
    test_images = os.path.join(DATASET_ROOT, "images")


    # 从注解文件中自动获取类别
    import json
    try:
        with open(train_json, 'r') as f:
            coco_data = json.load(f)

        # 提取类别信息
        categories = coco_data.get('categories', [])
        thing_classes = [cat['name'] for cat in categories]

        # 设置元数据
        for dataset_name in ["custom_train", "custom_val", "custom_test"]:
            if dataset_name in ['custom_train', 'custom_val', 'custom_test']:
                try:
                    MetadataCatalog.get(dataset_name).thing_classes = thing_classes
                except:
                    pass  # 如果数据集不存在就跳过

        print(f"类别数量: {len(thing_classes)}")
        print(f"类别名称: {thing_classes}")

        return len(thing_classes)

    except Exception as e:
        print(f"警告: 无法读取类别信息: {e}")
        # 使用默认类别
        thing_classes = ["bicycle", "chair", "diningtable", "bottle", "motorbike", "car", "tvmonitor", "bus"]  # 请根据实际情况修改

        for dataset_name in ["custom_train", "custom_val", "custom_test"]:
            try:
                MetadataCatalog.get(dataset_name).thing_classes = thing_classes
            except:
                pass

        return len(thing_classes)


if __name__ == "__main__":
    num_classes = register_custom_dataset()
    print(f"数据集注册完成！类别数量: {num_classes}")