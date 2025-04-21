# 查看信息
## 查看系统
```python
uname -o # 显示是 Linux 或 windows 或 macOS  
uname -m # 显示系统架构 如 x86_64 或 aarch64/arm64  
uname -o && uname -m # 合并命令
```  
本项目的系统为 `Linux x86_64`
## 查看显卡驱动
```python
nvidia-smi
```  
重点看Driver Version 和 CUDA Version
## 查看 conda 版本
```python
conda --version
```  
以下查看可选，可直接跳转到更新 conda 的步骤
## 查看是否安装了 cuda
```python
nvcc --version
```
## 查看 cuda 的安装路径
```python
which nvcc
```
## 查看是否安装了 cuDNN
### 如果已经查看了 cuda 的安装路径（一般类似于/usr/local/cuda/bin/nvcc）
优先查找较新的 cudnn_version.h (cuDNN 7+)：
```python
cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
```
如果上面的文件不存在或没有版本信息，尝试旧的 cudnn.h：
```python
cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2
```
如果都显示 No such file or directory，说明 cuDNN未安装在 /usr/local/cuda下
### 如果是在一个 conda 虚拟环境下
```python
conda list | grep cudnn  
```
如果这个命令有输出（显示 cudnn 包和版本号），那么 cuDNN 已经安装在你的 Conda 环境中了，只是没有安装在系统级的 /usr/local/cuda 路径下

# 更新 conda 
```python
conda update -n base -c defaults
```
`-n base`：指定更新基础（base）环境  
`-c defaults`：指定从默认（defaults）频道查找更新

# 清理 conda 缓存（可选）
```python
conda clean --all 或 conda clean --all -y  
```
# 配置 TensorFlow 环境
## 创建虚拟环境
官方推荐
```python
python3 -m venv tf-gpu python=<版本>  
```
功能强大（使用）
```python
conda create -n tf-gpu python=<版本>
```
> - tf-gpu为自定义的虚拟环境名称  
> - Note: Do not install TensorFlow with conda. It may not have the latest stable version. pip is recommended since TensorFlow is only officially released to PyPI.  
> - 本项目使用 `python=3.10.17` 的版本
## 进入虚拟环境
venv
```python
source tf-gpu/bin/activate   
```
conda
```python
    - conda activate tf-gpu
```
## 配置清华源
### 先查看通道配置
```python
conda config --show channels
```
### 清理旧配置
```python
conda config --remove-key channels
```
### 按以下顺序添加通道
```python
conda config --add channels defaults
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
```
## 安装 Tensorflow
### 更新 pip
```python
pip install --upgrade pip
```
### install tensorflow
For GPU users
```python
pip install tensorflow[and-cuda]
```
> - 指定 TensorFlow 版本：pip install 'tensorflow[and-cuda]==2.14.1'  
> - 如果在 Windows 上（命令提示符或 PowerShell）必须加引号 pip install 'tensorflow[and-cuda]'  
> - 更新 TensorFlow 版本：pip install 'tensorflow[and-cuda]' -U
For CPU users
```python
pip install tensorflow
```
> Note: Do not install TensorFlow with conda. It may not have the latest stable version. pip is recommended since TensorFlow is only officially released to PyPI.
## 验证 Tensorflow 是否安装成功
Verify the CPU setup:
```python
python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
```
If a tensor is returned, you've installed TensorFlow successfully.
Verify the GPU setup:
```python
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```
If a list of GPU devices is returned, you've installed TensorFlow successfully. If not continue to the next step.  

其他验证：  
```python
import tensorflow as tf  
print("TensorFlow Version: ", tf.__version__)  
print("CUDA Version Used by TensorFlow (Internal):", tf.sysconfig.get_build_info()["cuda_version"])  
print("cuDNN Version Used by TensorFlow (Internal):", tf.sysconfig.get_build_info()["cudnn_version"])  
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
```
如果GPU验证不成功：
- Create symbolic links to NVIDIA shared libraries:
```python
pushd $(dirname $(python -c 'print(__import__("tensorflow").__file__)'))
ln -svf ../nvidia/*/lib/*.so* .
popd
```
- Create a symbolic link to ptxas:
```python
ln -sf $(find $(dirname $(dirname $(python -c "import nvidia.cuda_nvcc;print(nvidia.cuda_nvcc.__file__)"))/*/bin/) -name ptxas -print -quit) $CONDA_PREFIX/bin/ptxas # conda环境
ln -sf $(find $(dirname $(dirname $(python -c "import nvidia.cuda_nvcc;print(nvidia.cuda_nvcc.__file__)"))/*/bin/) -name ptxas -print -quit) $VIRTUAL_ENV/bin/ptxas # venv创建的虚拟环境
```
- Verify the GPU setup:
```python
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```
- 重新安装 TensorFlow ，加上`-U`参数
## 忽略已经注册警告
```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=INFO, 1=WARNING, 2=ERROR, 3=FATAL 抑制警告信息，只显示错误或致命信息
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
```

# 更新 tensorflow-compression
原项目代码中的 tensorflow-compression 只适用于 macOS (见README.md) ，对于 Linux 系统需要重新安装，因为项目本身包含 tensorflow-compression 文件，所以需要**先删除项目文件中的 `tensorflow-compression`**，再重新安装
```python
rm -r /home/jay/workspace/adjscc/tensorflow_compression
python -m pip install tensorflow-compression
```
> 更新命令来自 https://github.com/tensorflow/compression
## 检验 tensorflow-compression 是否能够正常使用
```python
python -m tensorflow_compression.all_tests
```
> Once the command finishes, you should see a message OK (skipped=29) or similar in the last line.

# 其他
## 查看gpu使用情况
```python
pip install gpustat
gpustat -i
```
## 查看所有环境
```python
conda env list
```
## 退出虚拟环境
```python
conda deactivate
```
## 删除虚拟环境
```python
conda env remove -n <venv> # 官方推荐
conda remove --name <venv> --all
```
# 完整配置代码
```python
conda --version
conda update -n base -c defaults
conda clean --all -y
conda create -n tf-gpu python=3.10.17
conda activate tf-gpu
conda config --show channels
conda config --remove-key channels
conda config --add channels defaults
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
pip install --upgrade pip
pip install 'tensorflow[and-cuda]==2.14.1'
python3 -c "import tensorflow as tf; print(tf.sysconfig.get_build_info()['cuda_version'])"
python3 -c "import tensorflow as tf; print(tf.sysconfig.get_build_info()['cudnn_version'])"
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))" # 如果这里发现 GPU 验证不成功，回看前面的步骤
# 需要先删除项目文件中的 tensorflow-compression，否则会报错：RuntimeError: For tensorflow_compression, please install TensorFlow 2.1
rm -r /home/jay/workspace/adjscc/tensorflow_compression
python -m pip install tensorflow-compression
python -m tensorflow_compression.all_tests
```