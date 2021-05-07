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
```bash
# cmake命令
cmake .. -DPY_VERSION=3 -DWITH_ASCEND=OFF -DPYTHON_EXECUTABLE=`which python3`  -DPYTHON_INCLUDE_DIR=`python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())"` -DWITH_ARM=ON -DWITH_ASCEND_CL=ON -DWITH_TESTING=ON -DCMAKE_BUILD_TYPE=Release -DON_INFER=ON -DWITH_XBYAK=OFF
# make
make TARGET=ARMV8 -j$(nproc)
```
