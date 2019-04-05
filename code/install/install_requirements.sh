#!/bin/bash

# download and install annaconda2
wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda2-5.3.1-Linux-x86_64.sh
bash Anaconda2-5.3.1-Linux-x86_64.sh

# install tensorflow
pip install tensorflow-gpu==1.6.0

# install keras
pip install keras==2.2.4

# install pytorch
conda install pytorch-nightly cudatoolkit=9.0 -c pytorch

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
cd ..
