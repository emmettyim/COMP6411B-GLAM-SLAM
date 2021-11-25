import torch
import numpy as np
from PIL import Image
import cv2

if __name__ == '__main__':
    path = '/home/emmett/Documents/COMP6411B/GLAMpoints_KITTI/KITTI_00_gray/KITTI_00_gray.pt'
    
    model = torch.jit.load(path)
    model.eval()
    img = cv2.imread("000000.png",0)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img = cv2.resize(img, (256, 256))
    #img.transform.ToTensor();
    #img_norm = img.float()/float(image.max())
    #img_norm = img_norm.unsqueeze(0).unsqueeze(0).float()
    img_norm = np.float32(img) / float(np.max(img))
    img_norm = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0).float()
    img_norm = img_norm.cuda()
    output = model(img_norm)
    print(output)

