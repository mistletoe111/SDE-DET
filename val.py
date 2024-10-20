from ultralytics import YOLO
import os
from ultralytics import YOLO
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import random
import torch
def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
# 设置环境变量以避免 OpenMP 的警告
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def main():
    # 加载预训练模型
    model = YOLO(r"D:/桌面/训练数据/SDE-DET/weights/best.pt", task="detect")


    metrics = model.val(data="D:/桌面/ultralytics-main/youzi2.yaml")

    print(metrics)

if __name__ == '__main__':
    seed_torch(42)
    main()