# 算子性能优化 方法介绍

提供高性能的计算服务是飞桨的特色之一, 欢迎开发者为飞桨贡献高性能算子, 本文旨在向开发者提供一些快速实现高性能算子的方法。

# 基本介绍

- 算子性能优化工作的业务范围涵盖前向算子、反向算子、优化器等.

- 算子性能优化工作的基本目标是获得明显的算子性能提升, 力争达到业界一流的性能水平, 同时保证精度不会下降.

- 飞桨内算子性能优化主要围绕 GPU 计算开展, 因此需要用户掌握基本的[GPU 编程模型](https://developer.nvidia.com/zh-cn/blog/cuda-model-intro-cn/).


# 优化技巧

## 1.通用优化技巧

GPU Kernel 直接影响了算子性能, 我们推荐采用以下等通用优化策略提升 GPU Kernel 的性能, 从而削减算子的计算开销.

| 通用技巧 |
| -- |
| [向量化读写](<https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access>)|
| [协线程操作](<https://developer.nvidia.com/blog/cooperative-groups/>) |
| [Warp 级操作](<https://developer.nvidia.com/blog/using-cuda-warp-level-primitives>) |
| [共享内存操作](<https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/>) ([注意 Bank Conflicts](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/)) |


## 2. 飞桨内置优化技巧

我们在飞桨内开发并封装了一些优化技巧, 具体如下表所示, 欢迎使用, 也欢迎在使用过程中提出修改建议.

### 2.1 [线程配置优化](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/backends/gpu/gpu_launch_config.h)

我们推荐结合 OP 的使用场景设计对于的线程配置策略，如下图所示[IndexSample OP](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/index_sample_cn.html#index-sample)常用于处理 2 维数据, 因此使用[2 维的线程配置策略](https://github.com/PaddlePaddle/Paddle/blob/30838aa698d6f3f3b0860b052f6a50ef53ac6784/paddle/phi/kernels/gpu/index_sample_kernel.cu#L82-L91)相对比 1 维配置策略，性能可提升 20%左右。

<img src="https://raw.githubusercontent.com/PaddlePaddle/docs/develop/docs/dev_guides/images/index_sample.png" style="zoom:50%" />


优化 GPU Kernel 中的线程配置策略, 涵盖一维、二维、三维线程配置策略, 目前已经在`Elementwise`, `Stack`, `IndexSample`等 OP 中使用.

### 2.2 [Warp 计算优化](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/funcs/math_cuda_utils.h)

飞桨内对上文中提到的**Warp 级操作**进行了封装, 提供了简易的调用接口, 开发者可调用接口快速获得 Warp 内或者 Block 内的全部数据的求和、最大值、最小值.

<img src="https://raw.githubusercontent.com/PaddlePaddle/docs/develop/docs/dev_guides/images/cuda_math_utils.png" style="zoom:50%" />


### 2.3 [索引计算优化](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/funcs/fast_divmod.h):

当 GPU Kernel 的索引计算中存在除法或取模操作, 将在导致汇编层面计算开销变大, 我们建议采用快速除法优化这部分的计算开销。飞桨内[Pooling OP](https://github.com/PaddlePaddle/Paddle/blob/890c73158f663b327be7664ed6c4d08fb2c236a9/paddle/phi/kernels/funcs/pooling.cu#L41-L101) 采用索引优化计算后, 性能提升 1 倍.

<img src="https://raw.githubusercontent.com/PaddlePaddle/docs/develop/docs/dev_guides/images/fast_divmod.png" style="zoom:50%" />

### 2.4 [Kps 优化工具库](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/kernel_primitive_api/index_cn.html)

飞桨综合了一系列 GPU Kernel 通用性能优化技巧推出了 Kernel Primitive API，提供高性能的 Block 级 IO 运算和 Compute 运算。使用 Kernel Primitive API 进行 Kernel 开发可以更加专注计算逻辑的实现，在保证性能的同时大幅减少代码量，同时实现了算子计算与硬件解耦，详情见官网[Kernel Primitive API](https://www.paddlepaddle.org.cn/documentation/docs/zh/dev_guides/kernel_primitive_api/index_cn.html), 建议参考案例[ElementwiseAdd](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/kernel_primitive_api/add_example_cn.html)和[Reduce](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/kernel_primitive_api/reduce_example_cn.html) 使用。


### 3. C++模板特性

我们也鼓励充分挖掘 C++侧的可用优化点, 如使用`#pragma unroll`编译阶段加速指令，编译期自动展循环, 加速运行时循环的执行效率.

- 案例: [Elementwise_add OP](https://github.com/PaddlePaddle/Paddle/blob/30838aa698d6f3f3b0860b052f6a50ef53ac6784/paddle/phi/kernels/funcs/elementwise_base.h#L658-L661) 采用模板参数加速循环展开, 性能提升约 5%

```
struct SameDimsElementwisePrimitiveCaller {
  __device__ inline void operator()(Functor func, ArgsT *args, OutT *result) {
#pragma unroll
    for (int idx = 0; idx < VecSize; ++idx) {
      result[idx] = static_cast<OutT>(Apply(func, args[idx]));
    }
  }
};
```

### 4. 内置第三方库

飞桨内置了 cuBLAS, cuDNN, cuSOLVER, Thrust 等一系列第三方库, 若采用这些第三方等高性能计算库能获得显著的性能收益，也欢迎使用。cuBLAS 使用示例见[matmul_kernel_impl.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/impl/matmul_kernel_impl.h), cuDNN 的使用示例见[conv_kernel.cu](https://github.com/PaddlePaddle/Paddle/blob/30838aa698d6f3f3b0860b052f6a50ef53ac6784/paddle/phi/kernels/gpudnn/conv_kernel.cu#L366-L379), cuSOLVER 使用示例见[values_vectors_functor.h](https://github.com/PaddlePaddle/Paddle/blob/30838aa698d6f3f3b0860b052f6a50ef53ac6784/paddle/phi/kernels/funcs/values_vectors_functor.h#L219-L260), Thrust 使用示例见[coalesced_kernel.cu](https://github.com/PaddlePaddle/Paddle/blob/30838aa698d6f3f3b0860b052f6a50ef53ac6784/paddle/phi/kernels/sparse/gpu/coalesced_kernel.cu#L93-L106).
