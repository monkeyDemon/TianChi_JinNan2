# 环境安装

## 运行环境
Ubuntu 16.04

CUDA Version 9.0.176

CUDNN Version 7

NCCL Version 2.4.2

Python 2.7.15

Anaconda2-5.3.1

## 深度学习框架

程序同时使用二分类模型及目标检测模型进行违禁品的预测,其中:

目标检测模块使用pytorch及Detectron

分类模型使用keras

## 安装脚本

使用如下命令，将会自动安装除CUDA，CUDNN，及NCCL外的所有依赖环境及深度学习框架

```
$ cd /path/to/project/code/install
$ ./install_requirements.sh
```
该脚本将会依次执行如下操作：

下载并安装anaconda

下载并安装tensorflow-gpu==1.6

下载并安装keras==2.2.4

下载并安装pytorch

安装Detectron依赖项并编译

安装其他依赖
