import torch
import os
import cv2
from detectron2 import config
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import matplotlib.pyplot as plt
from detectron2.structures import Instances, Boxes

# 加载保存的模型
checkpoint = torch.load('saved_model.pth', map_location=torch.device('cpu'))  # 加载到 CPU

# 获取配置和权重
cfg = checkpoint['cfg']
model_state_dict = checkpoint['model_state_dict']

# 重新加载模型的权重
cfg.MODEL.DEVICE = 'cpu'  # 如果没有GPU，设置为 'cpu'

# 重新初始化模型
predictor = DefaultPredictor(cfg)
predictor.model.load_state_dict(model_state_dict)  # 加载保存的权重

# 图片路径
input_folder = 'input'
output_folder = 'result'
os.makedirs(output_folder, exist_ok=True)

# 指定要分割的类别 
target_classes = [39, 62, 64, 66]     # 39: bottle  62: TV  64: mouse  66: keyboard

# 获取所有图片文件
image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

# 遍历 input 文件夹中的图片
for img_name in image_files:
    
    # 读取图片
    img_path = os.path.join(input_folder, img_name)
    img = cv2.imread(img_path)

    # 预测
    outputs = predictor(img)

    # 获取预测的类别和框
    instances = outputs["instances"]
    pred_classes = instances.pred_classes  # 预测的类别
    pred_boxes = instances.pred_boxes  # 预测的框

    # 将 pred_classes 转换为 Tensor
    pred_classes = pred_classes.to(torch.int64)  # 确保它是 int64 类型

    # 使用 torch.isin 筛选目标类别
    target_classes_tensor = torch.tensor(target_classes, dtype=torch.int64)
    mask = torch.isin(pred_classes, target_classes_tensor)

    # 筛选出目标类别的预测框和掩膜
    filtered_instances = instances[mask]

    # 可视化
    v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]))
    out = v.draw_instance_predictions(filtered_instances)

    # 保存结果
    result_img_path = os.path.join(output_folder, img_name)
    cv2.imwrite(result_img_path, out.get_image()[:, :, ::-1])

    print(f"Processed {img_name} and saved the result to {result_img_path}")
    
print("All images processed and results saved!")

