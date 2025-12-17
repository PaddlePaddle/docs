# 使用指南

## 概述

随着大模型技术的快速发展，自定义算子（Custom Operator）已成为优化模型性能、扩展框架功能的关键手段。目前，PyTorch 生态中积累了大量高质量的自定义算子库和基于 Kernel DSL（如 Triton、TileLang）的算子实现。为了打破生态壁垒，帮助用户低成本地将这些优质算子资源迁移至 PaddlePaddle 框架，我们推出了一套跨生态自定义算子兼容机制。该机制支持用户在 PaddlePaddle 中直接使用 PyTorch 生态的自定义算子库和 Kernel DSL，从而大幅降低迁移成本，提升开发效率。

## 外部算子库

目前 PyTorch 生态中存在大量高质量的自定义算子库（如 FlashInfer、FlashMLA 等），这些算子库通常基于 CUDA/C++ 编写并封装为 Python 扩展。为了复用这些现有的算子库，我们提供了兼容性支持，使得用户可以直接在 PaddlePaddle 中安装并使用这些库，而无需进行繁琐的代码移植。

### 安装方式

对于使用基于兼容性方案的跨生态自定义算子库，一般情况分为两种安装方式：源码安装和 PyPI 安装。大部分算子库都托管在 GitHub 上，用户可以根据具体算子库的安装说明进行安装。下面以两个典型的算子库为例，介绍安装方式：

- 源码安装（以 `FlashInfer` 为例）：

    ```bash
    pip install paddlepaddle_gpu  # Install PaddlePaddle with GPU support, refer to https://www.paddlepaddle.org.cn/install/quick for more details
    git clone https://github.com/PFCCLab/flashinfer.git
    cd flashinfer
    git submodule update --init
    pip install apache-tvm-ffi>=0.1.2  # Use TVM FFI 0.1.2 or above
    pip install filelock jinja2  # Install tools for jit compilation
    # Install FlashInfer
    pip install --no-build-isolation . -v
    ```

- PyPI 安装（以 `TorchCodec` 为例）：

    ```bash
    pip install paddlecodec
    ```

个别算子库可能会有特殊的安装方式，请参考对应算子库 repo 中的 `README.md` 进行安装。

### 使用方式

安装完成后，即可在代码中直接导入并使用对应的算子库。为了实现跨生态兼容，用户需要在导入算子库之前，先启用 PaddlePaddle 的 PyTorch 代理层，以确保算子库中 `torch` 模块的调用能够正确映射到 `paddle` 模块。下面以 FlashMLA 为例说明使用方式：

```python
# 注意，在导入跨生态自定义算子库之前，需先启用 PaddlePaddle 的 PyTorch 代理层
# 即添加下面的两行
import paddle

# scope 为限定代理层生效的模块名称空间，避免影响其他模块的使用
paddle.enable_compat(scope={"flashinfer"})

# 之后即可导入并使用 flashinfer 库
import flashinfer
# 之后即可使用 flashinfer 下的算子，和 PyTorch 生态下的使用方式一致

# 下面以 flashinfer 中的 RMSNorm 算子为例
import numpy as np

def rms_norm(x, w, eps=1e-6):
    orig_dtype = x.dtype
    x = x.float()
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * paddle.rsqrt(variance + eps)
    x = x * w.float()
    x = x.to(orig_dtype)
    return x

batch_size = 99
hidden_size = 1024
dtype = paddle.float16

x = paddle.randn(batch_size, hidden_size).cuda().to(dtype)
w = paddle.randn(hidden_size).cuda().to(dtype)

y_ref = rms_norm(x, w)

y = flashinfer.norm.rmsnorm(x, w, enable_pdl=False)

# flashinfer 算子输出结果与参考实现保持一致
np.testing.assert_allclose(y_ref, y, rtol=1e-3, atol=1e-3)
```

### 已支持的算子库

PaddlePaddle 官方协同社区已经对社区中主流的跨生态自定义算子库进行了适配和测试，用户可以直接使用这些算子库而无需进行额外的修改。

我们将这些算子库统一放在组织 [PFCCLab](https://github.com/PFCCLab) 下，并列在下方。如果下方列表中没有你需要的算子库，可以移步至[原理和迁移方式](./design_and_migration_cn.md)，了解自定义算子兼容机制的实现原理，以及如何将你需要的算子库进行迁移。

以下是已经支持的跨生态自定义算子库列表：

| 算子库名称 | GitHub repo | PyPI 链接 |
| - | - | - |
| FlashInfer | [PFCCLab/flashinfer](https://github.com/PFCCLab/flashinfer) | - |
| FlashMLA | [PFCCLab/FlashMLA](https://github.com/PFCCLab/FlashMLA) | - |
| DeepGEMM | [PFCCLab/DeepGEMM](https://github.com/PFCCLab/DeepGEMM) | - |
| DeepEP | [PFCCLab/DeepEP](https://github.com/PFCCLab/DeepEP) | - |
| TorchCodec | [PFCCLab/paddlecodec](https://github.com/PFCCLab/paddlecodec) | [paddlecodec](https://pypi.org/project/paddlecodec/) |

## Kernel DSL

除去封装好的自定义算子库外，使用 Kernel DSL（Domain Specific Language，如 Triton、TileLang）直接编写算子也是一种常见的开发方式。这些 DSL 通常提供了更高级的抽象，使得用户能够更方便地编写高性能的算子。然而，这些 DSL 在实现中往往依赖于特定深度学习框架的状态管理接口，导致跨生态迁移困难。为此，我们也致力于提升这些新兴编写方式的跨生态兼容性，帮助用户更好地将其迁移至 PaddlePaddle 框架。

我们目前已经支持的 Kernel DSL 生态包括 [Triton](https://github.com/triton-lang/triton) 和 [TileLang](https://github.com/PFCCLab/tilelang-paddle)。

### 安装方式

```bash
# Triton 直接安装官方包即可
pip install triton
# TileLang 目前仍需要安装我们适配后的版本
pip install tilelang-paddle
```

### 使用方式

与其他自定义算子库相同，用户同样需要在导入对应的 Kernel DSL 库之前，先启用 PaddlePaddle 的 PyTorch 代理层。下面以 TileLang 为例说明使用方式：

```python
# 同样，在导入跨生态 Kernel DSL 库之前，需先启用 PaddlePaddle 的 PyTorch 代理层
import paddle

# 限定生效范围在 TileLang 模块
paddle.enable_compat(scope={"tilelang"})

import tilelang
import tilelang.language as T
import numpy as np

# 之后使用方式与官方 PyTorch 生态下保持一致
@tilelang.jit
def matmul(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float"):
    @T.prim_func
    def matmul_relu_kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        # Initialize Kernel Context
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            # Enable rasterization for better L2 cache locality (Optional)
            # T.use_swizzle(panel_size=10, enable=True)

            # Clear local accumulation
            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                # Copy tile of A
                # This is a sugar syntax for parallelized copy
                T.copy(A[by * block_M, ko * block_K], A_shared)

                # Copy tile of B
                T.copy(B[ko * block_K, bx * block_N], B_shared)

                # Perform a tile-level GEMM on the shared buffers
                # Currently we dispatch to the cute/hip on Nvidia/AMD GPUs
                T.gemm(A_shared, B_shared, C_local)

            # relu
            for i, j in T.Parallel(block_M, block_N):
                C_local[i, j] = T.max(C_local[i, j], 0)

            # Copy result back to global memory
            T.copy(C_local, C[by * block_M, bx * block_N])

    return matmul_relu_kernel

M = 1024
N = 1024
K = 1024
block_M = 128
block_N = 128
block_K = 32

# 定义并编译 Kernel 函数
matmul_relu_kernel = matmul(M, N, K, block_M, block_N, block_K)

# 创建随机输入张量
a = paddle.randn(M, K, device="cuda", dtype=paddle.float16)
b = paddle.randn(K, N, device="cuda", dtype=paddle.float16)
c = paddle.empty(M, N, device="cuda", dtype=paddle.float16)

# 运行 kernel
matmul_relu_kernel(a, b, c)

ref_c = paddle.nn.functional.relu(a @ b)

# 结果对齐
np.testing.assert_allclose(c.numpy(), ref_c.numpy(), rtol=1e-2, atol=1e-2)
```
