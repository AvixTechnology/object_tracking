# AI node

## 安裝环境

Jetson Orin Nano Version: 5.1.2
link is [here](https://developer.nvidia.com/embedded/jetpack)

```bash
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Sun_Oct_23_22:16:07_PDT_2022
Cuda compilation tools, release 11.4, V11.4.315
Build cuda_11.4.r11.4/compiler.31964100_0

```

ultralytics

```bash
pip install ultralytics
#tensorrt
sudo apt install python3-libnvinfer
#onnx
pip install onnx

sudo apt-get update
sudo apt-get upgrade
sudo apt-get install cmake
cmake --version # check the cmake version, this is only 3.16

#install cmake higher version from pip
pip install cmake --upgrade

pip install onnxsim==0.4.33 --user # this require cmake 3.22 or higher

#simple_pid
pip install simple_pid

```

onnx runtime要去網站下載
https://elinux.org/Jetson_Zoo#ONNX_Runtime

```cpp
wget -O onnxruntime_gpu-1.16.0-cp38-cp38-linux_aarch64.whl https://nvidia.box.com/shared/static/iizg3ggrtdkqawkmebbfixo7sce6j365.whl
pip install onnxruntime_gpu-1.16.0-cp38-cp38-linux_aarch64.whl
```

cuda torch 安裝

```bash
sudo apt-get -y update
sudo apt-get -y install autoconf bc build-essential g++-8 gcc-8 clang-8 lld-8 gettext-base gfortran-8 iputils-ping libbz2-dev libc++-dev libcgal-dev libffi-dev libfreetype6-dev libhdf5-dev libjpeg-dev liblzma-dev libncurses5-dev libncursesw5-dev libpng-dev libreadline-dev libssl-dev libsqlite3-dev libxml2-dev libxslt-dev locales moreutils openssl python-openssl rsync scons python3-pip libopenblas-dev

wget <https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl>
export TORCH_INSTALL=pytorch_whl/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl
python3 -m pip install --upgrade protobuf
python3 -m pip install --no-cache $TORCH_INSTALL

```

cuda torchvision 安裝

```bash
pip uninstall torchvision
sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev libopenblas-dev libavcodec-dev libavformat-dev libswscale-dev
git clone --branch v0.16.0 <https://github.com/pytorch/vision> torchvision
cd torchvision
export BUILD_VERSION=0.16.0
export MAX_JOBS=1
export FORCE_CUDA=1
python3 setup.py install --user  #會卡住 如果不設置MAX_JOBS

```

downgrade numpy to prevent error

```bash
python -m pip uninstall numpy
python -m pip install numpy==1.23.1

```

下載pt (運行以下python檔案)

```bash
from ultralytics import YOLO
import torch
print(torch.cuda.is_available())

model = YOLO('yolov8n.pt')

```

轉換成engine格式

```cpp
yolo export model=yolov8n.pt imgsz=480,640 device=0 format='engine'
```

# BotSort 
## install environment 
### git clone 
```
mkdir tracking_modules
cd tracking_modules
git clone https://github.com/NirAharon/BoT-SORT.git
```

### botsort 

```
cd BoT-SORT
pip install wheel
python3 -m pip install --upgrade setuptools
```

change the requirements.txt as followed
```
numpy
opencv-python
loguru
scikit-image
scikit-learn
tqdm
torchvision>=0.10.0
Pillow
thop
ninja
tabulate
tensorboard
# lap
motmetrics
filterpy
h5py
matplotlib
scipy
prettytable
# easydict
tensorboard
pyyaml
yacs
termcolor
gdown
# onnx==1.8.1
# onnxruntime==1.8.0
# onnx-simplifier==0.3.5
```
and then 

```
pip3 install -r requirements.txt
pip install lap
pip install easydict
pip install cmake
```

reboot the device!
```
reboot
```

```
sudo apt-get install libprotoc-dev protobuf-compiler
#pip install onnx
#pip install onnxruntime
pip install onnxsim
python3 setup.py develop
```
### for onnx 
```
wget https://nvidia.box.com/shared/static/iizg3ggrtdkqawkmebbfixo7sce6j365.whl -O onnxruntime_gpu-1.16.0-cp38-cp38-linux_aarch64.whl
pip3 install onnxruntime_gpu-1.16.0-cp38-cp38-linux_aarch64.whl
```

### other dependency 
```
pip3 install cython
pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

# Cython-bbox
pip3 install cython_bbox

pip install ultralytics

# UPDATE TORCH AND TORCHVISION FOR CUDA
wget https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl
export TORCH_INSTALL=~/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl
pip install --no-cache $TORCH_INSTALL

pip uninstall torchvision
sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev libopenblas-dev libavcodec-dev libavformat-dev libswscale-dev
git clone --branch v0.16.0 https://github.com/pytorch/vision torchvision
cd torchvision
export BUILD_VERSION=0.16.0
export MAX_JOBS=1
export FORCE_CUDA=1
python3 setup.py install   #會卡住 如果不設置MAX_JOBS
```

### numpy version check 
```
python -m pip uninstall numpy
python -m pip install numpy==1.23.1

export LD_PRELOAD='/home/avix/avix/tracking_modules/.venv/lib/python3.8/site-packages/scikit_learn.libs/libgomp-d22c30c5.so.1.0.0'

```


### Botsort pretrained folder 

#### create the folder in BotSort  and put the file to "pretrained " folder 

```
mkdir pretrained
```
https://drive.google.com/file/d/1uSmhXzyV1Zvb4TJJCzpsZOIcw7CCJLxj/view
https://drive.google.com/file/d/1QZFWpoa80rqo7O-HXmlss8J8CnS7IUsN/view


