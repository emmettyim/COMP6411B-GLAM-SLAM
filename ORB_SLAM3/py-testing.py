import torch
from models.glampoints import GLAMpoointsInference
from PIL import Image
import cv2

if __name__ == '__main__':
    kitti_model_path = '/home/emmett/Documents/COMP6411B/GLAMpoints_KITTI/'
    trained_sequence = 'KITTI_10_gray' #train on kitti 00 02 03 05 06 07 08 10
    model_name = '/model_best.pth'
    model_path = kitti_model_path + trained_sequence + model_name
    print("Model Path:")
    print(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #model setting
    nms = 10 # Value of the NMS window applied on the score map output of GLAMpointsInference (default:10)
    min_prob = 0.0 #Minimum probability of a keypoint for GLAMpointsInference (default:0)

    with torch.no_grad():
        glampoints = GLAMpointsInference(path_weights=model_path, nms=nms, min_prob=min_prob)
        model = glampoints.net

        #example = torch.rand(1, 1, 376, 1241) #dim of input tensor 1x1xHxW = size of kitti image
        #example = example.to(device)

        # traced_model = torch.jit.trace(model, example)
        img = cv2.imread("cat.png")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img = cv2.resize(img, (256, 256))
        output = model(img)
        print(output[0][:2])

