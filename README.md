Machine Learning for Human Pose Estimate
================

## Essential python model
* [Numpy](http://www.numpy.org) for a highly optimized library for numerical operations with a MATLAB-style syntax.
* [Chainer](http://chainer.org) neural network (deep learning) framework
* [opencv3.0](http://opencv.org) OpenCV-Python for solve computer vision problem

## Using linear regression to predict the joints
- Datasets: [subset](http://cims.nyu.edu/~tompson/flic_plus.htm) of full [FLIC](http://vision.grasp.upenn.edu/cgi-bin/index.php?n=VideoLearning.FLIC)
- There are five different approach:
	1. Use all the pixels of transformed pictures as inputs and SGD -> `python my_scripts/train.py`
	2. Use all the pixels of transformed pictures as inputs and BGD -> `python my_scripts/train.py --mode bgd`
	3. Use SIFT based on Bag-of-Word algorithm and SGD -> `python my_scripts/train_sift.py`
	4. Use pixels of patch images around the joints and SGD -> `python my_scripts/train_bgd.py`
- Training results are in [Notebook](training_results.ipynb)

## Reference
- [Deeppose](https://github.com/mitmul/deeppose)
- [Stanford on line course](http://cs231n.github.io)
