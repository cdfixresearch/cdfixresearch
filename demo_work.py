from main import demo_work
import numpy as np
import os.path as osp
import torch
import os
from main import Tree


data = torch.load(osp.join("D:\\research\\auto_fix\\cdfxicode\\processed", 'data_2.pt'))
dic = np.load(osp.join("D:\\research\\auto_fix\\cdfxicode\\processed", 'dic.npy'))
demo_work(data, dic)
