
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
# Load a model
model = YOLO(r"/root/ultralytics-main/ultralytics/da.yaml")  # build a new model from scratch
#model = YOLO(r"D:\ultralytics-main\runs\detect\train17\weights\best.pt)  # load a pretrained model (recommended for training)
if __name__ == '__main__':
    seed_torch(42)
# Use the model
    model.train(data="/root/ultralytics-main/youzi2.yaml", pretrained=False,epochs=100,batch=8,workers=8)  # train the model