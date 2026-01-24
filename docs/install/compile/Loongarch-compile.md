# 龙芯架构 PaddlePaddle 源码编译指南

## 概述

本文介绍如何在龙芯（Loongarch）架构下从源码编译安装 PaddlePaddle。以 Loongson-3C6000 处理器为例进行说明。

## 环境准备

### 硬件与系统要求

| 项目 | 版本/型号 |
|------|-----------|
| 处理器 | Loongson-3C6000 |
| 操作系统 | Loongnix Server release 23.1 |
| Python | 3.11.6 (64 bit) |
| pip/pip3 | 23.3.1 (64 bit) |

### 预装环境说明

龙芯操作系统 Loongnix Server release 23.1 默认安装了以下工具：

- **GCC**: 12.3.0
- **CMake**: 3.26.3
- **Python**: 3.11.6
- **pip**: 23.3.1

## 安装步骤

> **注意**：目前在龙芯处理器和龙芯国产化操作系统上安装 Paddle，仅支持源码编译方式。

### 1. 安装依赖包

安装系统依赖库和 Python wheel：

```bash
yum install libffi-devel openssl-devel libsqlite3x-devel sqlite-devel \
    ghc-lzma tk uuid gdbm-devel gdbm openjpeg2-devel zlib-devel \
    libjpeg-turbo-devel openjpeg2 patch patchelf

pip3 install wheel
```

### 2. 下载源码

克隆 PaddlePaddle 源码仓库：

```bash
git clone https://github.com/PaddlePaddle/Paddle.git
cd Paddle
```

### 3. 切换到发布分支

切换到 release/3.3 分支进行编译：

```bash
git checkout release/3.3
```

### 4. 安装 Python 依赖库

根据 `requirements.txt` 文件安装所需的 Python 依赖库：

```bash
pip3 install -r requirements.txt
```

### 5. 创建编译目录

```bash
mkdir build && cd build
```

### 6. 调整系统限制

链接过程中会打开大量文件，需要调整系统限制以避免编译错误：

```bash
ulimit -n 4096
```

### 7. 执行 CMake 配置

```bash
cmake .. \
    -DWITH_LOONGARCH=ON \
    -DWITH_TESTING=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DON_INFER=ON \
    -DWITH_XBYAK=OFF \
    -DWITH_MKL=OFF
```

**配置参数说明**：

- `WITH_LOONGARCH=ON`: 启用龙芯架构支持
- `WITH_TESTING=OFF`: 关闭测试
- `CMAKE_BUILD_TYPE=Release`: 发布版本编译
- `ON_INFER=ON`: 启用推理模式
- `WITH_XBYAK=OFF`: 关闭 XBYAK
- `WITH_MKL=OFF`: 关闭 MKL

### 8. 编译

使用所有可用 CPU 核心进行并行编译：

```bash
make -j$(nproc)
```

> **提示**：编译过程可能需要较长时间，请耐心等待。

### 9. 安装编译产物

编译成功后，进入 `Paddle/build/python/dist` 目录，找到生成的 `.whl` 安装包。

在当前机器或目标机器上安装：

```bash
python3 -m pip install -U paddlepaddle-*.whl
```

或者：

```bash
python -m pip install -U paddlepaddle-*.whl
```

## 验证安装

安装完成后，验证 PaddlePaddle 是否安装成功：

```bash
python3 << EOF
import paddle
paddle.utils.run_check()
EOF
```

或者进入 Python 交互式环境：

```bash
python3
```

然后执行：

```python
import paddle
paddle.utils.run_check()
```

如果输出以下信息，说明安装成功：

```
PaddlePaddle is installed successfully!
```

## 常见问题

### 编译过程中出现文件打开数限制错误

**解决方案**：执行 `ulimit -n 4096` 增加文件打开数限制。

### 依赖包安装失败

**解决方案**：检查网络连接，或使用国内镜像源：

```bash
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

## 总结

恭喜！至此您已完成 PaddlePaddle 在龙芯环境下的源码编译和安装。

---
