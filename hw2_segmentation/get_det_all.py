import torch
import os
import cv2
from detectron2 import config
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import matplotlib.pyplot as plt

# 加载保存的模型
checkpoint = torch.load('saved_model.pth')

# 获取配置和权重
cfg = checkpoint['cfg']
model_state_dict = checkpoint['model_state_dict']

# 重新加载模型的权重
cfg.MODEL.DEVICE = 'cpu'  # 如果没有GPU，设置为 'cpu'

# 重新初始化模型
predictor = DefaultPredictor(cfg)
predictor.model.load_state_dict(model_state_dict)  # 加载保存的权重

# 输入图片路径
input_folder = 'input'  # 假设图片在 'input' 文件夹中
output_folder = 'result'  # 保存结果到 'result' 文件夹

# 创建 'result' 文件夹（如果不存在）
os.makedirs(output_folder, exist_ok=True)

# 获取所有图片文件（1.jpg, 2.jpg, 3.jpg,...）
image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

# 遍历图片进行推理和保存结果
for img_name in image_files:
    img_path = os.path.join(input_folder, img_name)

    # 读取图片
    image = cv2.imread(img_path)

    # 使用模型进行推理
    outputs = predictor(image)

    # 获取分割结果（Mask）
    v = Visualizer(image[:, :, ::-1], metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=0.8)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    # 获取输出结果的图片并保存
    output_img = v.get_image()[:, :, ::-1]  # 转回BGR格式（OpenCV）
    result_img_path = os.path.join(output_folder, img_name)  # 保存到 'result' 文件夹

    # 保存结果
    cv2.imwrite(result_img_path, output_img)

    print(f"Processed {img_name} and saved the result to {result_img_path}")

print("All images processed and results saved!")
