import torch
from detectron2 import config
from detectron2.engine import DefaultPredictor
import os

# 配置文件和权重文件
config_file = "./detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
weights_file = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"

# 加载配置
cfg = config.get_cfg()
cfg.merge_from_file(config_file)
cfg.MODEL.WEIGHTS = weights_file
cfg.MODEL.DEVICE = 'cpu'  # 如果没有GPU，设置为 'cpu'

# 初始化预测器
predictor = DefaultPredictor(cfg)

# 保存整个模型（包括权重和配置）
torch.save({
    'cfg': cfg,  # 保存配置
    'model_state_dict': predictor.model.state_dict()  # 保存模型的权重
}, 'saved_model.pth')

print("模型已保存！")