#!/bin/bash
# install wget
apt install wget
apt install lrzsz

# install cudnn
tar -xvf cudnn-9.0-linux-x64-v7.3.1.20.tgz
sudo cp cuda/include/cudnn.h /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*

# install ncll
sudo dpkg -i nccl-repo-ubuntu1604-2.4.2-ga-cuda9.0_1-1_amd64.deb 
sudo apt update
sudo apt install libnccl2=2.4.2-1+cuda9.0 libnccl-dev=2.4.2-1+cuda9.0

# download and install annaconda2
wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda2-5.3.1-Linux-x86_64.sh
bash Anaconda2-5.3.1-Linux-x86_64.sh -y

cp cuda/lib6/libcudnn.so.7 ~/anaconda2/lib

# install tensorflow
pip install tensorflow-gpu==1.6.0

# install keras
pip install keras==2.2.4

# install pytorch
conda install pytorch-nightly cudatoolkit=9.0 -c pytorch
conda install pytorch-nightly-1.0.0.dev20190315-py2.7_cuda9.0.176_cudnn7.4.2_0.tar.bz2

# install cocoapi
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make install
python setup.py install --user

# active Detectron 
pip install -r ../Detectron/requirements.txt
cd ../Detectron
make

# install zlib
tar -xvf ./zlib-1.2.9.tar.gz
cd zlib-1.2.9
sudo -s
./configure; make; make install
cd /lib/x86_64-linux-gnu
ln -s -f /usr/local/lib/libz.so.1.2.9/lib libz.so.1
