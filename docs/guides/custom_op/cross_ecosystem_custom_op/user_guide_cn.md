# 使用指南

## 概述

随着大模型的兴起，在深度学习框架之上构建自定义算子（Custom Operator）已成为提升模型性能和功能的关键手段。而目前 PyTorch 作为深度学习领域的主流框架之一，拥有大量的自定义算子实现。为了帮助用户更好地将现有的 PyTorch 等生态的自定义算子迁移至 PaddlePaddle 框架，我们提供了自定义算子兼容机制，旨在降低迁移成本，提升开发效率。

## 使用步骤

### 一般安装方式

对于使用基于兼容性方案的跨生态自定义算子库，一般情况下只需要 clone 后通过 pip 安装对应的算子库即可使用。下面以 FlashMLA 为例说明安装方式：

```bash
git clone https://github.com/PFCCLab/FlashMLA.git
cd FlashMLA
pip install .
```

对于部分已经发布到 PyPI 的自定义算子库，也可以直接通过 pip 安装。下面以 TorchCodec 为例：

```bash
pip install paddlecodec
```

个别算子库可能会有特殊的安装方式，请参考对应算子库的说明文档进行安装。

### 使用方式

安装完成后，即可在代码中直接导入并使用对应的算子库。为了实现跨生态兼容，用户需要在导入算子库之前，先启用 PaddlePaddle 的 PyTorch 代理层，以确保算子库中 `torch` 模块的调用能够正确映射到 `paddle` 模块。下面以 FlashMLA 为例说明使用方式：

```python
import paddle

paddle.compat.enable_torch_proxy({"flash_mla"})

import flash_mla
# 之后即可使用 flash_mla 下的算子
```

## 已支持的算子库

PaddlePaddle 官方协同社区已经对社区中主流的跨生态自定义算子库进行了适配和测试，用户可以直接使用这些算子库而无需进行额外的修改。

我们将这些算子库统一放在组织 [PFCCLab](https://github.com/PFCCLab) 下，并列在下方。如果下方列表中没有你需要的算子库，可以移步至[原理和迁移](./design_cn.md)，了解自定义算子兼容机制的实现原理，以及如何将你需要的算子库进行迁移。

### FlashInfer

#### 安装方式

#### 使用方式

### FlashMLA

### DeepGEMM

### DeepEP

### TorchCodec

## 已支持的 Kernel DSL 库

### TileLang

### Triton
