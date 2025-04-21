# 使用到的网页
安装tensorflow：https://tensorflow.google.cn/install/pip
安装compression：https://github.com/tensorflow/compression

# 查看信息
## 查看系统
uname -o 显示是 Linux 或 windows 或 macOS
uname -m 显示系统架构 如 x86_64 或 aarch64/arm64
uname -o && uname -m

## 查看显卡驱动
nvidia-smi
重点看Driver Version 和 CUDA Version

## 查看 conda 版本
conda --version

## 查看是否安装了 cuda
nvcc --version

## 查看 cuda 的安装路径
which nvcc

## 查看是否安装了 cuDNN
### 如果已经查看了 cuda的安装路径（一般类似于/usr/local/cuda/bin/nvcc）
优先查找较新的 cudnn_version.h (cuDNN 7+)：
cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
如果上面的文件不存在或没有版本信息，尝试旧的 cudnn.h：
cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2
如果都显示 No such file or directory，说明 cuDNN未安装在 /usr/local/cuda下
### 如果是在一个 conda 虚拟环境下
conda list | grep cudnn
如果这个命令有输出（显示 cudnn 包和版本号），那么 cuDNN 已经安装在你的 Conda 环境中了，只是没有安装在系统级的 /usr/local/cuda 路径下

# 更新 conda 
conda update -n base -c defaults
-n base：指定更新基础（base）环境
-c defaults：指定从默认（defaults）频道查找更新

# 清理 conda 缓存
conda clean --all 
或
conda clean --all -y

# 创建 conda 虚拟环境
官方推荐：python3 -m venv tf-gpu python=3.10
或
功能强大：conda create -n tf-gpu python=3.10
（tf-gpu为自定义的虚拟环境名称；python=3.11为指定的python版本）

# 退出虚拟环境
conda deactivate

# 更新pip
pip install --upgrade pip

# 配置清华源
## 先查看通道配置
conda config --show channels
## 清理旧配置
conda config --remove-key channels
## 按以下顺序添加通道
conda config --add channels defaults
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
## 安装 cuda、cuDNN、TensorFlow
conda install -c conda-forge cudatoolkit=12.2 cudnn=8.9
pip install tensorflow==2.15.0
或
conda install -c conda-forge tensorflow=2.15.0=cuda120py311h5cbd639_2
python3 -m pip install 'tensorflow[and-cuda]' -U
**python3 -m pip install 'tensorflow[and-cuda]==2.14.1'**

### 检验是否安装成功
import tensorflow as tf
print("TensorFlow Version: ", tf.__version__)
print("CUDA Version Used by TensorFlow (Internal):", tf.sysconfig.get_build_info()["cuda_version"]) # 可能显示构建时使用的版本，如 '12.2'
print("cuDNN Version Used by TensorFlow (Internal):", tf.sysconfig.get_build_info()["cudnn_version"]) # 可能显示构建时使用的版本，如 '8' 或 '8.9'
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

### 忽略已经注册警告
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=INFO, 1=WARNING, 2=ERROR, 3=FATAL 抑制警告信息，只显示错误或致命信息
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# 运行命令
## 运行adjscc_cifar10.py（需要指定参数）
python adjscc_cifar10.py train -h
'-h'：打印出脚本的用法说明、可接受的参数选项及它们的含义
python adjscc_cifar10.py train -ct awgn（指定了模式'train'，信道类型'awgn'）
## 指定预训练模型路径
python adjscc_cifar10.py eval -ct awgn --load_model_path /path/to/your/model.h5

# 更新tensorflow-compression
python -m pip install tensorflow-compression -U

# 查看gpu使用情况
pip install gpustat
gpustat -i

# 查看环境
conda env list

# 删除虚拟环境
conda env remove -n <venv>
conda remove --name <venv> --all