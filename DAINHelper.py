import torch
import numpy as np
from networks.DAIN_xx import DAIN
# from networks.DAIN import DAIN

# from AverageMeter import *

torch.backends.cudnn.benchmark = True

class DAIN2X:
    def __init__(self, weights_path: str) -> None:
        # torch.set_default_dtype(torch.float16)
        self.model = DAIN().type(torch.get_default_dtype())
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        self.model.load_state_dict(torch.load(weights_path))
        self.model.eval()
    
    def __call__(self, frame) -> np.ndarray:
        with torch.no_grad():
            return self.model.processRgb(frame)
