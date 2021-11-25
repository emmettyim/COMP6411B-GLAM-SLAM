#include <torch/script.h>
#include <torch/torch.h>
#include <vector>
#include <iostream>
#include <memory>

#include "GLAMpoints.h"

using namespace cv;
using namespace std;

// should I use network or just use the results qaq cry

namespace ORB_SLAM3
{
//below should only run once -> import once in 
/**    torch::jit::script::Module module;
    try{
        mdodule = torch::jit::load(..\modelpath);
    }
    catch(const c10::Error& e){
        std::cerr << "error loading the model\n"
    }**/

int GLAMpoints::operate()( InputArray _image, vector<KeyPoint>&_keypoints, OOutputArray_descriptors, std::vector<int> &vLappingArea)
{
    if(_image.empty())
        return -1;
    // get image and change it to tensor from opencv mat? treat it as input to model

    
}
    void ComputeKeyPoints(InputArray _image,std::vector<std::vectoor<KeyPoint>> &_keypoints){
        Mat image; //Gray scales
        Mat readImage = _image.getMat();
        //cv::cvtColor(readImage, image, cv::COLOR_GRAY2RGB); //maybe not needed? try
        assert(readImage.type() == CV_8UC1);

        //Preprocess
        //not sure if need to resize
        //cv::Size scale(256,256)；
        //cv::resize(readImage, readImage, scale, 0, 0, cv::INTER_LINEAR);

        //if weird stuff happens check this later, {1dim, 1 channel, size}
        torch::Tensor tensor_image = torch::from_blob(readImage.data, {1, readImage.rows, readImage.cols, 1},torch::kFloat32);
        tensor_image = tensor_image.permute({0,3,1,2});

        // transforms.Normalize(mean=[0.485, 0.456, 0.406],
        //                    std=[0.229, 0.224, 0.225]) below not sure
        // tensor_image[0][0] = tensor_image[0][0].sub_(0.485).div_(0.229);
        // tensor_image[0][1] = tensor_image[0][1].sub_(0.456).div_(0.224);
        // tensor_image[0][2] = tensor_image[0][2].sub_(0.406).div_(0.225);

        tensor_image_norm = tensor_image / (torch::max(tensor_image));
        tensor_image_norm = tensor_image_norm.unsqueeze(0).unsqueeze(0).toType(torch::kFloat32);
        tensor_image_norm = tensor_image_norm.to(torch::kCUDA);
        torch::Tensor output = module.forward({tensor_image}).toTensor();

        std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/2) << '\n';

        cv::Mat outputKp = torchTensortoCVMat(output);

        int nkeypoints = 0;
        nkeypoints = outputKp.size().height;
        
        _keypoints = vector<cv::KeyPooint>(nkeypoints);

        int i = 0;        
        for (vector<KeyPoint>::iterator keypoint = keypoints.begin(), 
                    keypointEnd = keypoints.end(); keypoint != keypointEnd; ++keypoint){
                        Point2i pt;
                        pt.x = outputKp.at<int>(i,0);
                        pt.y = outputKp.at<int>(i,1);
                        i++;
                        keypoint->pt = pt;
                        keypoint->size = 10;
                    }
    }


    cv::Mat torchTensortoCVMat(torch::Tensor& tensor){
        //tensor = tensor.squeeze().detach();
        //tensor = tensor.permute({1, 2, 0}).contiguous();
        //tensor = tensor.mul(255).clamp(0, 255).to(torch::kU8);
        //tensor = tensor.to(torch::kCPU);
        int64_t height = tensor.size(0);
        int64_t width = tensor.size(1);
        cv::Mat mat = cv::Mat(cv::Size(width, height), CV_32S, tensor.data_ptr<uchar>());
        return mat.clone();

        /** or below
         * tensor = tensor.to(torch::kCPU);
         * cv::Mat mat(cv::Size(width,height), CV_32F, i_tensor.data_ptr());
         * 
         * return mat;
         * 
         **/
    }

} //namespace ORB_SLAM

using namespace cv;
using namespace std;