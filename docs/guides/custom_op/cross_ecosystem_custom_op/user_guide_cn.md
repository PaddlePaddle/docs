# 使用指南

## 概述

随着大模型的兴起，在深度学习框架之上构建自定义算子（Custom Operator）已成为提升模型性能和功能的关键手段。而目前 PyTorch 作为深度学习领域的主流框架之一，拥有大量的自定义算子实现。为了帮助用户更好地将现有的 PyTorch 等生态的自定义算子迁移至 PaddlePaddle 框架，我们提供了自定义算子兼容机制，旨在降低迁移成本，提升开发效率。

## 使用步骤

### 安装方式

对于使用基于兼容性方案的跨生态自定义算子库，一般情况下只需要 clone 后通过 pip 安装对应的算子库即可使用。下面以 `FlashInfer` 为例说明安装方式：

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

对于部分已经发布到 PyPI 的自定义算子库，也可以直接通过 pip 安装。下面以 `TorchCodec` 为例：

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
paddle.compat.enable_torch_proxy(scope={"flashinfer"})

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

## 已支持的算子库

PaddlePaddle 官方协同社区已经对社区中主流的跨生态自定义算子库进行了适配和测试，用户可以直接使用这些算子库而无需进行额外的修改。

我们将这些算子库统一放在组织 [PFCCLab](https://github.com/PFCCLab) 下，并列在下方。如果下方列表中没有你需要的算子库，可以移步至[原理和迁移方式](./design_and_migration_cn.md)，了解自定义算子兼容机制的实现原理，以及如何将你需要的算子库进行迁移。

以下是已经支持的跨生态自定义算子库列表：

| 算子库名称 | GitHub repo | PyPI 链接 |
| - | - | - |
| FlashInfer | [PFCCLab/flashinfer](https://github.com/PFCCLab/flashinfer) | - |
| FlashMLA | [PFCCLab/FlashMLA](https://github.com/PFCCLab/FlashMLA) | - |
| DeepGEMM | [PFCCLab/DeepGEMM](https://github.com/PFCCLab/DeepGEMM) | - |
| DeepEP | [PFCCLab/DeepGEMM](https://github.com/PFCCLab/DeepEP) | - |
| TorchCodec | [PFCCLab/paddlecodec](https://github.com/PFCCLab/paddlecodec) | [paddlecodec](https://pypi.org/project/paddlecodec/) |

## Kernel DSL 生态支持

除去自定义算子外，编写自定义算子的方式也在不断演进，涌现出了诸如 Kernel DSL（如 Triton、TileLang）等新兴的编写方式。这些新兴的编写方式在实现中往往或多或少依赖于特定深度学习框架的状态管理接口，从而导致跨生态迁移的难度加大。为此，我们也致力于提升这些新兴编写方式的跨生态兼容性，帮助用户更好地将其迁移至 PaddlePaddle 框架。

我们目前已经支持的 Kernel DSL 生态包括 Triton 和 TileLang。安装方式分别如下：

```bash
# Triton 直接安装官方包即可
pip install triton
# TileLang 目前仍需要安装我们适配后的版本
pip install tilelang-paddle
```

与其他自定义算子库相同，用户同样需要在导入对应的 Kernel DSL 库之前，先启用 PaddlePaddle 的 PyTorch 代理层。下面以 TileLang 为例说明使用方式：

```python
# 同样，在导入跨生态 Kernel DSL 库之前，需先启用 PaddlePaddle 的 PyTorch 代理层
import paddle

# 限定生效范围在 TileLang 模块
paddle.compat.enable_torch_proxy(scope={"tilelang"})

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
