"""
* This file is part of implementing GLAMpoints with pySLAM
*
* For COMP6411B project use
"""
import sys 
import math 
from enum import Enum
import numpy as np
from skimage.feature import peak_local_max 
import cv2
import torch
from PIL import Image


# https://www.robots.ox.ac.uk/~vgg/publications/2012/Arandjelovic12/arandjelovic12.pdf
# adapated from https://www.pyimagesearch.com/2015/04/13/implementing-rootsift-in-python-and-opencv/

def non_max_suppression(image, size_filter, proba):
    non_max = peak_local_max(image, min_distance=size_filter, threshold_abs=proba, \
                      exclude_border=True, indices=False)
    kp = np.where(non_max>0)
    if len(kp[0]) != 0:
        for i in range(len(kp[0]) ):

            window=non_max[kp[0][i]-size_filter:kp[0][i]+(size_filter+1), \
                           kp[1][i]-size_filter:kp[1][i]+(size_filter+1)]
            if np.sum(window)>1:
                window[:,:]=0
    return non_max
    
def center_crop(img, size):
    """
    Get the center crop of the input image
    Args:
        img: input image [HxWx3]
        size: size of the center crop (tuple)
    Output:
        img_pad: center crop
        x, y: coordinates of the crop
    """

    if not isinstance(size, tuple):
        size = (size, size)

    img = img.copy()
    h, w = img.shape[:2]

    pad_w = 0
    pad_h = 0
    if w < size[1]:
        pad_w = np.uint16((size[1] - w) / 2)
    if h < size[0]:
        pad_h = np.uint16((size[0] - h) / 2)
    img_pad = cv2.copyMakeBorder(img,
                                 pad_h,
                                 pad_h,
                                 pad_w,
                                 pad_w,
                                 cv2.BORDER_CONSTANT,
                                 value=[0, 0, 0])
    x1 = w // 2 - size[1] // 2
    y1 = h // 2 - size[0] // 2

    img_pad = img_pad[y1:y1 + size[0], x1:x1 + size[1]]

    return img_pad

class GLAMpoints2D:
    def __init__(self, feature):
        # initialize the SIFT feature detector
        self.feature = feature
        self.nms = 10
        self.min_prob = 0.0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Change path below for geting corresponding pytorch model
        self.path = '/home/emmett/Documents/COMP6411B/GLAMpoints_KITTI/KITTI_00_gray/KITTI_00_gray.pt'
        self.model = torch.jit.load(self.path)
        self.model.eval()

    
    def pre_process_data(self, image):
        '''
        pre-process the image by first putting it in the right format 1x1xHxW and by dividing it by max(image)
        :param image:
        :return:
        '''
        ## Change to gray image
        #image_gray = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
        # Perform central crop and padding to 256 * 256 (suggested by GLAM)
        #print(PIL.__version__)
        #print(type(image1))
        #print(image1.shape)
        #image = image1
        #image = center_crop(image1,256)
        if torch.is_tensor(image):
            # if it is already a Torch Tensor
            if len(image.shape) == 2:
                # gray image
                image_norm = image.float() / float(image.max())
                image_norm = image_norm.unsqueeze(0).unsqueeze(0).float().to(self.device)
            if len(image.shape) == 3:
                # BxHxW, HxWx3 or 3xHxW
                if image.shape[0] == 3:
                    # 3xHxW
                    image_numpy = image.permute(1,2,0).cpu().numpy() # now HxWx3
                    image = cv2.cvtColor(image_numpy, cv2.COLOR_RGB2GRAY)
                    image_norm = np.float32(image) / float(np.max(image))
                    # creates a Torch input tensor of dimension 1x1xHxW
                    image_norm = torch.from_numpy(image_norm).unsqueeze(0).unsqueeze(0).float().to(self.device)
                elif image.shape[2] == 3:
                    # HxWx3
                    image_numpy = image.cpu().numpy()
                    image = cv2.cvtColor(image_numpy, cv2.COLOR_RGB2GRAY)
                    image_norm = np.float32(image) / float(np.max(image))
                    # creates a Torch input tensor of dimension 1x1xHxW
                    image_norm = torch.from_numpy(image_norm).unsqueeze(0).unsqueeze(0).float().to(self.device)
                else:
                    # BxHxW, already gray scale
                    image_norm = image.float() / float(image.max())
                    image_norm = image_norm.unsqueeze(1).float().to(self.device)
            if len(image.shape) == 4:
                # Bx1xHxW or BxHxWx1 or Bx3xHxW or BxHxWx3
                if image.shape[1] == 1:
                    # already good Bx1xHxW
                    image_norm = image.float() / float(image.max())
                elif image.shape[3] == 1:
                    # BxHxWx1
                    image_norm = image.permute(0,3,1,2).float() / float(image.max())
                else:
                    if image.shape[1] == 3:
                        # Bx3xHxW
                        image = image.permute(0, 2, 3, 1)
                    # now tensor BxHxWx3
                    B,H,W,_ = image.shape
                    image_gray = torch.Tensor((), dtype=torch.float32)
                    image_gray.new_zeros((B,H,W))
                    for i in image.shape[0]:
                        image_numpy_i = image[i].cpu().numpy()
                        image_i = cv2.cvtColor(image_numpy_i, cv2.COLOR_RGB2GRAY)
                        image_norm_i = np.float32(image_i) / float(np.max(image))
                        image_gray[i] = image_norm_i

                    image_norm = image_gray.unsqueeze(1).float().to(self.device)

        elif isinstance(image, type(np.empty(0))):
            if len(image.shape) != 2:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image_norm = np.float32(image) / float(np.max(image))

            # creates a Torch input tensor of dimension 1x1xHxW
            image_norm = torch.from_numpy(image_norm).unsqueeze(0).unsqueeze(0).float().to(self.device)
        return image_norm.float().to(self.device)


    def detect(self, frame, mask=None):
        # Preprocess images
        image = self.pre_process_data(frame)
        # Import image to model
        output = self.model(image)
        # compute non-max-suppression output
        output_nms = non_max_suppression(output.data.cpu().numpy().squeeze(), self.nms, self.min_prob)
        # Obtain keypoints and descriptors
        kp_map = np.where(output_nms > 0)
        kp_array = np.array([kp_map[1],kp_map[0]]).T
        kp_cv2 = [cv2.KeyPoint(kp_array[i,0], kp_array[i,1], 10) for i in range(len(kp_array))]
        (kp,des) = self.feature.compute(np.uint8(frame), kp_cv2)

        return des
 
    def transform_descriptors(self, des, eps=1e-7): 
        # apply the Hellinger kernel by first L1-normalizing and 
        # taking the square-root
        des /= (des.sum(axis=1, keepdims=True) + eps)
        des = np.sqrt(des)        
        return des 
            
    def compute(self, frame, kps, eps=1e-7):
        # Preprocess images
        image = self.pre_process_data(frame)
        # Import image to model
        output = self.model(image)
        # compute non-max-suppression output
        output_nms = non_max_suppression(output.data.cpu().numpy().squeeze(), self.nms, self.min_prob)
        # Obtain keypoints and descriptors
        kp_map = np.where(output_nms > 0)
        kp_array = np.array([kp_map[1],kp_map[0]]).T
        kp_cv2 = [cv2.KeyPoint(kp_array[i,0], kp_array[i,1], 10) for i in range(len(kp_array))]
        (kps,des) = self.feature.compute(np.uint8(frame), kp_cv2)

        ## compute SIFT descriptors
        #(kps, des) = self.feature.compute(frame, kps)

        # if there are no keypoints or descriptors, return an empty tuple
        if len(kps) == 0:
            return ([], None)

        # apply the Hellinger kernel by first L1-normalizing and 
        # taking the square-root
        # this step is for root-sift, can be tested on GLAM
        des = self.transform_descriptors(des)

        # return a tuple of the keypoints and descriptors
        return (kps, des)

    # detect keypoints and their descriptors
    # out: kps, des 
    def detectAndCompute(self, frame, mask=None):    	
        # Preprocess images
        image = self.pre_process_data(frame)
        # Import image to model
        output = self.model(image)
        print(output.data.cpu().numpy().squeeze())
        # compute non-max-suppression output
        output_nms = non_max_suppression(output.data.cpu().numpy().squeeze(), self.nms, self.min_prob)
        # Obtain keypoints and descriptors
        kp_map = np.where(output_nms > 0)
        kp_array = np.array([kp_map[1],kp_map[0]]).T
        
        kp_cv2 = [cv2.KeyPoint(float(kp_array[i,0]), float(kp_array[i,1]), 10.0) for i in range(len(kp_array))]
        (kps,des) = self.feature.compute(np.uint8(frame), kp_cv2)
        #kps = np.array([m.pt for m in kps], dtype=np.int32)
        # if there are no keypoints or descriptors, return an empty tuple
        if len(kps) == 0:
            return ([], None)

        # apply the Hellinger kernel by first L1-normalizing and 
        # taking the square-root
        # this step is for root-sift, can be tested on GLAM
        des = self.transform_descriptors(des)

        # return a tuple of the keypoints and descriptors
        return (kps, des)
