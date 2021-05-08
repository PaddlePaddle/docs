# 飞桨框架NPU版安装说明


## 安装方式：通过release/2.1分支源码编译安装

### 环境准备

**昇腾NPU**

- **处理器: 鲲鹏920**
- **操作系统：Linux version 4.19.36-vhulk1907.1.0.h475.eulerosv2r8.aarch64**
- **CANN: 20.3**
- **Python版本： 2.7/3.6/3.7 (64 bit)**
- **pip或pip3版本：9.0.1+ (64 bit)**
- **cmake版本：3.15+**
- **gcc/g++版本：8.2+**


### 源码编译安装步骤：


**第一步**：下载Paddle源码并编译，CMAKE编译选项含义请参见[编译选项表](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/Tables.html#Compile)

```bash
# 下载源码，建议切换到 develop 分支
git clone -b develop https://github.com/PaddlePaddle/Paddle.git
cd Paddle

# 创建编译目录
mkdir build && cd build

# 执行cmake
cmake .. -DPY_VERSION=3 -DWITH_ASCEND=OFF -DPYTHON_EXECUTABLE=`which python3`  -DPYTHON_INCLUDE_DIR=`python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())"` -DWITH_ARM=ON -DWITH_ASCEND_CL=ON -DWITH_TESTING=ON -DCMAKE_BUILD_TYPE=Release -DON_INFER=ON -DWITH_XBYAK=OFF

# 使用以下命令来编译
make TARGET=ARMV8 -j$(nproc)
```
**第二步**：安装与验证编译生成的wheel包

编译完成之后进入`Paddle/build/python/dist`目录即可找到编译生成的.whl安装包，推荐设置PYTHONPATH环境变量安装：

```bash
# 设置PYTHONPATH
export PYTHONPATH=/your/dir/Paddle/build/python:$PYTHONPATH

# 验证命令
python -c "import paddle; paddle.utils.run_check()"
```
