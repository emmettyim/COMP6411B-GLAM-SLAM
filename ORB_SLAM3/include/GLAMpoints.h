/**
* This file is part of ORB-SLAM3 introoduced by COMP6411B Group 9 in introducing
* a  new hybrid SLAM model
* 
* Further description to be added
**/

#ifndef GLAMPOINTS_H
#define GLAMPOINTS_H

#include <vector>
#include <list>
#include <opencv2/opencv.hpp>

#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include <memory>

namespace ORB_SLAM3
{

class GLAMpoints
{
public:    
    GLAMpoints();
    ~GLAMpoints(){}

    int GLAMpoints::operate()( InputArray _image, vector<KeyPoint>&_keypoints, OOutputArray_descriptors, std::vector<int> &vLappingArea);


  protected:

    cv::Mat torchTensortoCVMat(torch::Tensor& tensor); 
    void ComputeKeyPoints(std::vector<std::vectoor<KeyPoint>> &_keypoints);

}
} //namespace ORB_SLAM
#endif