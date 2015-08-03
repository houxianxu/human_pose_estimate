#! /bin/bash
if [ ! -d data ]; then
    mkdir data
fi
cd data

# get FLIC-full dataset and FLIC-Plus annotations
wget http://vision.grasp.upenn.edu/video/FLIC-full.zip
unzip FLIC-full.zip
rm -rf FLIC-full.zip
cd FLIC-full
wget http://cims.nyu.edu/~tompson/data/tr_plus_indices.mat
cd ..
