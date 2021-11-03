import torch
from torch.autograd import Variable
from torchvision import transforms as T
import cv2
import numpy as np

# predict single image (40x40, single channel), 
# default device is GPU!
def pred_single_img(curr:np.array, next:np.array, net, in_channels:int=2, device:str='cuda:0'):
    net.eval()
    to_tensor = T.ToTensor()
    curr = to_tensor(curr)
    next = to_tensor(next)
    if in_channels==2:
        curr = torch.cat((curr, next), 0)
    curr = curr[None,...]
    input = Variable(curr.to(device=device))
    pred = net(input)
    out_put = pred.detach().cpu().numpy().argmax()
    return out_put


# curr_path = '/content/COMP9517_Project/Sequences_p/mitosis_classification/normal_cell/01_127/curr/(1002, 411)_0.jpg'
# next_path = '/content/COMP9517_Project/Sequences_p/mitosis_classification/normal_cell/01_127/next/(1002, 411)_0.jpg'

curr_path = '/content/COMP9517_Project/Sequences_p/mitosis_classification/mitosis_tra_pos/01/curr/(362, 153)_1.jpg'
next_path = '/content/COMP9517_Project/Sequences_p/mitosis_classification/mitosis_tra_pos/01/next/(362, 153)_1.jpg'

curr = cv2.imread(curr_path, 0)
next = cv2.imread(next_path, 0)

weights = '/content/checkpoints_mitosis/mitosis_best_model.pt'
net = torch.load(weights)

print(pred_single_img(curr,next,net))