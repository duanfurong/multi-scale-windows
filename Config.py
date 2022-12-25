import torch

n_epochs = 500
NUM_CLASSES = 7
ACTIVITY_SIZE = 2004
seed = 2
variance = [0.1, 0.1]
CENTER_VARIANCE = 0.1
CONFIDENCE_THRESHOLD = 0.05
MAX_PER_IMAGE = 100
MAX_PER_CLASS = -1
NMS_THRESHOLD = 0.45
clip = 0.35
model_path = 'model.ckpt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
ACTIVITY_LABEL = ['DownStairs','Jogging','Sitting','Standing','UpStairs', 'Walking']
TITLE = 'WISDM Confusion Matrix'