# COMP6411B-GLAM-SLAM
A Hybrid SLAM pipeline with GLAMpoints detector -- based on pySLAM
-----------------------------------


This project aims at building a hybrid Visual SLAM system with GLAMpoints based on existing SIFT-SLAM using pySLAM. The result of GLAM-SLAM is compared with other modern SLAM methods and the result is satisfactory.

**Some results are illustrates below:**
ORB-SLAM2 is run with c++ ORB-SLAM2 pipeline (with relocalization and loop closure) while GLAM-SLAM is run with pySLAM(without relocalization and loop closure) 

*GLAMpoints matching with KITTI-00*
![match](https://user-images.githubusercontent.com/17170219/143461341-49ee46f6-50d1-4fe6-a8e0-90a502e8b52b.png)

----

*Running GlAM-SLAM with KITTI-06*\
<img width="386" alt="GLAM-SLAM1" src="https://user-images.githubusercontent.com/17170219/143460689-16814c8b-24eb-4d27-8dbe-a1311fd39acf.png">

----

*KITTI-06 & KITTI-10 ORB-SLAM2 and GLAM-SLAM comparison, TUM-G stands for GLAM-SLAM in TUM style*
<p float="left">
<img width="386" alt="KITTI-06 Comparison" src="https://user-images.githubusercontent.com/24664133/143476833-5cbe3f16-421f-44bc-a480-b7f93833fe34.png">
<img width="386" alt="KITTI-06 Comparison - xyz" src="https://user-images.githubusercontent.com/24664133/143476837-c44f2230-4aa4-40f4-853f-64585c1e044f.png">
<img width="386" alt="KITTI-10 Comparison" src="https://user-images.githubusercontent.com/24664133/143476839-a5d4a326-bfe0-449e-ae1f-a6bc14b6d8ae.png">
<img width="386" alt="KITTI-10 Comparison - xyz" src="https://user-images.githubusercontent.com/24664133/143476841-415f0b1a-407f-47a4-bcff-dec1073fe2a2.png">
</p>

----

*GLAM-SLAM on KITTI-06 and KITTI-10*
<p float="left">
  <img width="386" alt="KITTI-06 ARE" src="https://user-images.githubusercontent.com/24664133/143469595-38591378-44e1-4b3a-907f-b0a3b542db96.png">
  <img width="386" alt="KITTI-06 RPE" src="https://user-images.githubusercontent.com/24664133/143469602-4c461c97-a12b-45f5-9624-7abb39ac7fe2.png">
  <img width="386" alt="KITTI-06 Tra" src="https://user-images.githubusercontent.com/24664133/143469606-df9196f6-a8dc-43c4-bb0c-e175c69d2eaf.png">

  <img width="386" alt="KITTI-10 ARE" src="https://user-images.githubusercontent.com/24664133/143469613-67a37f83-7156-46b4-8ea0-2d605f58a7fb.png">
  <img width="386" alt="KITTI-10 RPE" src="https://user-images.githubusercontent.com/24664133/143469617-17b34d9a-b1ac-4ebf-905c-2a113a600f92.png">
  <img width="386" alt="KITTI-10 Tra" src="https://user-images.githubusercontent.com/24664133/143469620-90924728-9e19-4792-94c9-eb625f15c836.png">
</p>

----


In order to compile the project, one will need to have build an environment for pySLAM to run, please follow the below instructions/refer to orginal repo (see reference belows:

The framework has been developed and tested under **Ubuntu 18.04**.  
A specific install procedure is available for: 
- [Ubuntu 20.04](#install-pyslam-under-ubuntu-2004)
- [MacOs](#install-pyslam-on-macos) 
- [Windows](https://github.com/luigifreda/pyslam/issues/51)

Clone this repo and its modules by running 
```
$ git clone --recursive https://github.com/luigifreda/pyslam.git
```
then run

`$ ./install_all.sh`

Once you have run the script `install_all.sh` (as required [above](#requirements)), you can test  `main_slam.py` by running:

```
$ python3 -O main_slam.py
```
You can change the 'config.init' and main_slam.py to select other detectors.
Available detectors and descriptors includes:

At present time, the following feature **detectors** are supported: 
* *[FAST](https://www.edwardrosten.com/work/fast.html)*  
* *[Good features to track](https://ieeexplore.ieee.org/document/323794)* 
* *[ORB](http://www.willowgarage.com/sites/default/files/orb_final.pdf)*  
* *[ORB2](https://github.com/raulmur/ORB_SLAM2)* (improvements of ORB-SLAM2 to ORB detector) 
* *[SIFT](https://www.cs.ubc.ca/~lowe/papers/iccv99.pdf)*   
* *[SURF](http://people.ee.ethz.ch/~surf/eccv06.pdf)*   
* *[KAZE](https://www.doc.ic.ac.uk/~ajd/Publications/alcantarilla_etal_eccv2012.pdf)*
* *[AKAZE](http://www.bmva.org/bmvc/2013/Papers/paper0013/paper0013.pdf)* 
* *[BRISK](http://www.margaritachli.com/papers/ICCV2011paper.pdf)*  
* *[AGAST](http://www.i6.in.tum.de/Main/ResearchAgast)*
* *[MSER](http://cmp.felk.cvut.cz/~matas/papers/matas-bmvc02.pdf)*
* *[StarDector/CenSurE](https://link.springer.com/content/pdf/10.1007%2F978-3-540-88693-8_8.pdf)*
* *[Harris-Laplace](https://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/mikolajczyk_ijcv2004.pdf)* 
* *[SuperPoint](https://github.com/MagicLeapResearch/SuperPointPretrainedNetwork)*
* *[D2-Net](https://github.com/mihaidusmanu/d2-net)*
* *[DELF](https://github.com/tensorflow/models/tree/master/research/delf)*
* *[Contextdesc](https://github.com/lzx551402/contextdesc)*
* *[LFNet](https://github.com/vcg-uvic/lf-net-release)*
* *[R2D2](https://github.com/naver/r2d2)*
* *[Key.Net](https://github.com/axelBarroso/Key.Net)*
* *[DISK](https://arxiv.org/abs/2006.13566)*
* *[GLAMpoints](https://arxiv.org/pdf/2104.00099.pdf)*

The following feature **descriptors** are supported: 
* *[ORB](http://www.willowgarage.com/sites/default/files/orb_final.pdf)*  
* *[SIFT](https://www.cs.ubc.ca/~lowe/papers/iccv99.pdf)*
* *[ROOT SIFT](https://www.robots.ox.ac.uk/~vgg/publications/2012/Arandjelovic12/arandjelovic12.pdf)*
* *[SURF](http://people.ee.ethz.ch/~surf/eccv06.pdf)*    
* *[AKAZE](http://www.bmva.org/bmvc/2013/Papers/paper0013/paper0013.pdf)* 
* *[BRISK](http://www.margaritachli.com/papers/ICCV2011paper.pdf)*     
* *[FREAK](https://www.researchgate.net/publication/258848394_FREAK_Fast_retina_keypoint)* 
* *[SuperPoint](https://github.com/MagicLeapResearch/SuperPointPretrainedNetwork)*
* *[Tfeat](https://github.com/vbalnt/tfeat)*
* *[BOOST_DESC](https://www.labri.fr/perso/vlepetit/pubs/trzcinski_pami15.pdf)*
* *[DAISY](https://ieeexplore.ieee.org/document/4815264)*
* *[LATCH](https://arxiv.org/abs/1501.03719)*
* *[LUCID](https://pdfs.semanticscholar.org/85bd/560cdcbd4f3c24a43678284f485eb2d712d7.pdf)*
* *[VGG](https://www.robots.ox.ac.uk/~vedaldi/assets/pubs/simonyan14learning.pdf)*
* *[Hardnet](https://github.com/DagnyT/hardnet.git)*
* *[GeoDesc](https://github.com/lzx551402/geodesc.git)*
* *[SOSNet](https://github.com/yuruntian/SOSNet.git)*
* *[L2Net](https://github.com/yuruntian/L2-Net)*
* *[Log-polar descriptor](https://github.com/cvlab-epfl/log-polar-descriptors)*
* *[D2-Net](https://github.com/mihaidusmanu/d2-net)*
* *[DELF](https://github.com/tensorflow/models/tree/master/research/delf)*
* *[Contextdesc](https://github.com/lzx551402/contextdesc)*
* *[LFNet](https://github.com/vcg-uvic/lf-net-release)*
* *[R2D2](https://github.com/naver/r2d2)*
* *[BEBLID](https://raw.githubusercontent.com/iago-suarez/BEBLID/master/BEBLID_Boosted_Efficient_Binary_Local_Image_Descriptor.pdf)*
* *[DISK](https://arxiv.org/abs/2006.13566)*

### Evaluate with evo
First, move generated trajectories evo/project_eva,
change the KITTI groundtruth to TUM format together with corresponding timestamp with evo contri script **kitti_poses_and_timestamps_to_trajectory.py** 
```
./kitti_poses_and_timestamps_to_trajectory.py ../project_eva/(input_groundtruth).txt ../project_eva/(groundtruth_timestamp).txt ../project_eva/(output_groundtruth).txt
```
Evaluate with evo using TUM evaluation
```
evo_ape tum (output_groundtruth).txt (SLAM_output).txt -va --plot -plot_mode xz --align --correct_scale --save_results results/00_ORB.zip
```
For multiple trajectories:
```
evo_traj tum (SLAM_output1).txt (SLAM_output2).txt --ref=(output_groundtruth).txt -p --plot_mode=xz --align --correct_scale
```
See evo Github for more
[evaluation of odometry and SLAM](https://github.com/MichaelGrupp/evo)


### Directory
- KITTI : ```~/Documents/COMP6411B/KITTI```
- GITHUB: ```~/Documents/GITHUB```
-----------------------------------
### Reference
##### Project details and baseline
- [Project Proposal](https://docs.google.com/document/d/1VT4LuWXs3p1wdCg1wgFVtcjLj78ZTWRq_PWc5E-sGaw/edit)
- [Paper :GLAMpoints-SLAM](https://arxiv.org/pdf/2104.00099.pdf)
- [Github:GLAMpoints_pytorch](https://github.com/PruneTruong/GLAMpoints_pytorch)
- [Github:PySLAM](https://github.com/luigifreda/pyslam)

##### References
- [Github:Dataset List](https://github.com/youngguncho/awesome-slam-datasets#urban)
- [Github:DBoW2](https://github.com/dorian3d/DBoW2)
- [Paper :Training Orientation Estimators](https://openaccess.thecvf.com/content_cvpr_2016/papers/Yi_Learning_to_Assign_CVPR_2016_paper.pdf)
- [Paper :Training Descriptors](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4269996)
- [Paper :LIFT](https://link.springer.com/content/pdf/10.1007%2F978-3-319-46466-4_28.pdf)
- [Github:LIFT(tensorflow)](https://github.com/cvlab-epfl/tf-lift)
- [Github:GLAMpoints_pytorch](https://github.com/PruneTruong/GLAMpoints_pytorch)
- [Gitlab:GLAMpoints_tf(not optimized)](https://gitlab.com/retinai_sandro/glampoints/-/tree/master)
- [Github:PySLAM](https://github.com/luigifreda/pyslam)
