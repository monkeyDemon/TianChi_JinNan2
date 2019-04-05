# 使用说明

# 一、算法概述

算法的核心模型为二分类resnet50模型和目标检测模型retinanet。

二分类resnet50网络我们使用keras进行训练。

目标检测网络retinanet我们使用FaceBook开源的Detectron框架进行训练。

通过二分类网络对待诊断图像是否包含违禁品进行过滤，从而有效滤除明显不包含违禁品但却杂乱的图像，从而降低后一阶段目标检测网络的虚警率。

最终使用目标检测网络对图像中包含的违禁品的种类及位置进行预测。


# 二、安装

运行如下命令，将会自动安装除CUDA，CUDNN，及NCCL外的所有依赖环境及深度学习框架
```
$ cd /path/to/project/code/install
$ ./install_requirements.sh
```
详见`project/code/README.md`

# 三、推理（提交结果复现）

最终提交的`json`文件在`project/submit/json`目录下保存, 其中`2019-03-26-21-40-53.json`是初赛B榜的最终提交版本。

复现`2019-03-26-21-40-53.json`文件需要两个模型权重文件，分别是：

`project/code/model/object_detection_model`下的目标检测模型权重文件`model_final.pkl`

和`/project/code/model/cnn_model`下的二分类CNN模型权重文件`resnet50_classify2_sigmoid_final.h5`

提交的代码文件中已经包含了上述的两个训练好的模型文件，在第四部分`训练`中将会详细介绍如何复现训练的过程。

下面介绍初赛B榜结果的推理过程

### 3.1 预处理
首先解压出B榜测试集`jinnan2_round1_test_b_20190326`，放置于`project/data/First_round_data`下

然后对B榜测试集进行预处理，运行如下脚本：
```
project/code/shell$ ./pre_process_sigmoid.sh
```
该脚本读取`project/data/First_round_data/jinnan2_round1_test_b_20190326`下的数据，并对图片进行了一种类似于sigmoid函数的映射,我们认为这有助于提升模型性能。预处理后的图片存储在`project/data/ProcessData/jinnan2_round1_test_b_20190326_sigmoid`中，这个过程可能会提示一些文件夹不存在的错误，是因为没有解压其他数据造成，对推理过程不产生影响。

### 3.2 推理二分类模型预测结果

运行如下脚本：
```
project/code/classify_keras$ ./second_stage_inference_classify_result.sh
```
该脚本将会加载训练好的二分类CNN模型，对测试集中每张图片是否为限制品进行推理，结果记录在`project/data/ProcessData/cnn_judgement_result.json`中

### 3.3 推理目标检测模型得到最终结果

运行如下脚本：
```
project/code/shell$ ./inference.sh
```
该脚本将会加载训练的目标检测模型，结合二分类模型的预测结果，在`project/submit/json`下按照运行时间产出最终的json文件，例如`2019-03-27-17-07-48.json`


# 四、训练（训练过程复现）

## 复现模型的流程大致为

1.数据准备，将比赛数据置于 project/data/First_round_data下

2.下载预训练模型

3.于 project/code/shell 和 project/code/classify_keras 目录下运行shell脚本，完成各种预处理工作及模型的训练。

4.按照第三节所述进行模型推理得到最终结果。

### 4.1 数据准备

将比赛数据置于 project/data/First_round_data下，并解压

### 4.2 下载预训练模型

下载目标检测网络RetinaNet的预训练模型，置于`project/code/model/pre_train_model`下
https://dl.fbaipublicfiles.com/detectron/ImageNetPretrained/MSRA/R-50.pkl

下载CNN分类模型resnet50预训练模型，置于`project/code/model/pre_train_model`下
https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5

### 4.3 模型训练

全程分为三个过程，预处理阶段、训练阶段和预测阶段，其中训练阶段细分为如下的三个阶段。

**预处理阶段**

依次运行如下脚本：

```
project/code/shell$ ./pre_process_sigmoid.sh
project/code/shell$ ./pre_process_extract_restricted.sh
project/code/shell$ ./pre_process_modified_model.sh
project/code/shell$ ./pre_process_get_null_json.sh
```

**训练第一阶段**

为了权衡模型的precision和recall，第一阶段将会训练一个最基础的模型，该模型用于筛选出容易被误认为包含违禁品的正常图片。将这些图片加入后续的训练中将有助于降低模型虚警率。

依次运行如下脚本：

划分出训练集和验证集
```
project/code/shell$ ./first_stage_split_dataset.sh
```

进行第一阶段训练
```
project/code/shell$ ./first_stage_train.sh
```

推断出容易被误认为包含违禁品的正常图片
```
project/code/shell$ ./first_stage_inference_normal.sh
project/code/shell$ json_path='json_file_name.json'
project/code/shell$ ./first_stage_split_normal.sh $json_path  
```
其中，`first_stage_inference_normal.sh`脚本将会在`project/submit/json`下生成一个json标注文件，由于该文件按照运行时间命名，因此需要手动指定json的文件名。

将`json_file_name.json`替换为实际产生的json文件的文件名。

例如：json_path='2019-03-28-16-16-56.json'

**训练第二阶段**

依次运行如下脚本：

```
project/code/shell$ ./second_stage_generate_data.sh
project/code/shell$ ./second_stage_split_dataset.sh
project/code/shell$ ./second_stage_train.sh
```

训练二分类模型，依次运行如下脚本（注意切换路径）：
```
project/code/classify_keras$ ./second_stage_split_dataset.sh
project/code/classify_keras$ ./second_stage_train_classify_model.sh
project/code/classify_keras$ ./second_stage_inference_classify_result.sh
```


**训练第三阶段**

依次运行如下脚本：
```
project/code/shell$ ./third_stage_data_augmentation.sh
project/code/shell$ ./third_stage_generate_diffcult_data.sh
project/code/shell$ ./third_stage_split_dataset.sh
project/code/shell$ ./third_stage_train.sh
project/code/shell$ ./third_stage_draw_learning_curve.sh
```
