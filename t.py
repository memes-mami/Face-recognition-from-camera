import torch
from super_gradients.training import models

model = models.get("yolo_nas_m", pretrained_weights="coco")
torch.save(model.state_dict(), "weights/yolo_nas_m_coco.pt")
