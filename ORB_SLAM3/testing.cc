#include "torch/script.h"
#include "torch/torch.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <memory>

using namespace std;
using namespace cv;

cv::Mat torchTensortoCVMat(torch::Tensor& tensor){
    
        tensor = tensor.to(torch::kCPU).squeeze();

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

int main()
{

    torch::jit::script::Module module;
    try{
        module = torch::jit::load("KITTI_00_gray.pt");
    }
    catch(const c10::Error&){
        std::cerr << "error loading the model\n";
        return -1;
    }

    model.to(at::kCPU);

        Mat image = cv::imread("000000.png",0); //Gray scales
        Mat readImage = _image.getMat();
        //cv::cvtColor(readImage, image, cv::COLOR_BGA2GRAY); //maybe not needed? try
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
        std::cout << _leypoints[1] << '\n';

    return 0;
}