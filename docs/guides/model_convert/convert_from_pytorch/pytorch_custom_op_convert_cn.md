# Pytorch 自定义算子转写教程

迁移 Pytorch 自定义算子可以借鉴 torch 的实现代码，在 paddle 和 torch 的不同之处做一些修改即可。修改实现代码中 Pytorch 的 api 为 paddle 的 api，可参考[Pytorch 与 paddle api 映射表](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/model_convert/pytorch_api_mapping_cn.html#api)

## 迁移自定义 C++算子

迁移自定义 C++算子注意事项：

1. 复用 Pytorch 的 kernel 实现代码
2. 引入 paddle 扩展头文件 ```#include "paddle/extension.h"```
3. 修改实现代码中涉及到 Pytorch 的代码
  - 3.1. Pytorch 的 tensor 修改为 paddle 定义的 tensor：``` paddle::Tensor```
  - 3.2. paddle Place 的使用：```paddle::GPUPlace()```，详见[paddle 自定义 c++算子文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/custom_op/new_cpp_op_cn.html#shebeileixing)
  - 3.3. paddle 支持的 tensor API，如：empty、full、empty_like、full_like、DataType 等；详见[Tensor API](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/custom_op/new_cpp_op_cn.html#tensor-api)
  - 3.4. 算子实现中需包含前向(forward)实现, 如果需要包括算子梯度计算则需要包含反向(backward)实现, 实现中需要注意:
    - 3.4.1. paddle::Tensor 需要以 ```const paddle::Tensor&``` 的形式作为输入
    - 3.4.2. 返回值只能是```std::vector<paddle::Tensor> ```
    - 3.4.3. Attribute 仅支持特定数据类型，详见[运算函数与基础 API](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/custom_op/new_cpp_op_cn.html#api)
  - 3.5. 静态图模式下实现中需包含前向维度推导（InferShape）和类型推导（InferDtype）的函数，详见[维度与类型推导函数](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/custom_op/new_cpp_op_cn.html#weiduyuleixingtuidaohanshushixian)
4. 构建算子
  - 4.1. ```PD_BUILD_OP```：用于构建前向算子
    - 4.1.1. 包含```Inputs()```,``` Attrs()```, ```Outputs()```, ```SetKernelFn()```,``` SetInferShapeFn()```,``` SetInferDtypeFn()``` 参数指定
  - 4.2. ```PD_BUILD_GRAD_OP``` ：用于构建前向算子对应的反向算子
    - 4.2.1. 包含```Inputs()```,``` Attrs()```,``` Outputs()```,``` SetKernelFn()``` 参数指定
  - 4.3. 详见[构建算子](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/custom_op/new_cpp_op_cn.html#goujiansuanzi)

5. 使用 setuptools 编译
  - 5.1. 详见[setuptools 编译](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/custom_op/new_cpp_op_cn.html#setuptools)

其他：可参考[paddle 自定义算子文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/custom_op/new_cpp_op_cn.html)

## 迁移自定义 python 算子

迁移自定义 python 算子注意事项：

1. 复用 Pytorch 自定义算子实现，并将算子实现中使用到的 torch api 改为对应的 paddle api, 参考[Pytorch 与 paddle api 映射表](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/model_convert/pytorch_api_mapping_cn.html#api)
2. Pytorch 中自定义 op 继承自```torch.autograd.Function```改为 paddle 中的```paddle.autograd.PyLayer```
3. 定义算子的 forward 和 backward 方法
   - 3.1. ```forward()```和```backward()```定义为 staticmethod
   - 3.2. 第一个参数是 PyLayerContext 对象 ctx
   - 3.3. forward 和 backward 中如果需要传递信息，可以通过在 forward 中使用```ctx.save_for_backward```保存 tensor 信息，在 backward 中通过```ctx.saved_tensor```读取
4. 通过调用算子的类方法```apply()```实现算子调用

详见[paddle 动态图自定义 Python 算子文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/custom_op/new_python_op_cn.html#id2)

## 自定义算子转写示例

### 自定义 c++算子

以 3D 检测模型中用到的 bev pool v2 算子为例，转写过程参考如下

#### 接入算子定义文件

自定义算子通常通过定义.cc 文件调用对应的 kernel 实现，从而在网络的前反向中使用对应的算子。算子定义文件的要求可参考[Paddle 官方文档自定义算子部分](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/custom_op/new_cpp_op_cn.html)

以 bev pool v2 为例，算子定义文件需包含以下步骤：

__1. 引入 paddle 自定义算子头文件 <paddle/extension.sh>__

```c++
#include <paddle/extension.h>
```

__2. 声明 kernel 实现文件中定义的 kernel 调用函数__
注意与 kernel 实现文件保持一致，如 bev pool v2:

```c++
// CUDA function declarations
void bev_pool_v2(int c, int n_intervals, const float* depth, const float* feat,
    const int* ranks_depth, const int* ranks_feat, const int* ranks_bev,
    const int* interval_starts, const int* interval_lengths, float* out);

void bev_pool_v2_grad(int c, int n_intervals, const float* out_grad,
  const float* depth, const float* feat, const int* ranks_depth, const int* ranks_feat,
  const int* ranks_bev, const int* interval_starts, const int* interval_lengths,
  float* depth_grad, float* feat_grad);
```

__3. 实现算子前向计算函数__
  - 3.1. 接入算子 kernel 函数

  算子的前向计算函数的核心是算子 kernel 函数，自定义算子接入时应先确定算子是否有已经实现的 kernel 代码，如果是接入 torch 中已经存在的算子，通常 torch 模型中已经有实现好的 kernel 代码，如 bev pool v2 的 kernel 代码如下

```c++
//bev pool v2 前向 kernel 实现
__global__ void bev_pool_v2_kernel(int c, int n_intervals,
                                  const float *__restrict__ depth,
                                  const float *__restrict__ feat,
                                  const int *__restrict__ ranks_depth,
                                  const int *__restrict__ ranks_feat,
                                  const int *__restrict__ ranks_bev,
                                  const int *__restrict__ interval_starts,
                                  const int *__restrict__ interval_lengths,
                                  float* __restrict__ out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int index = idx / c;
  int cur_c = idx % c;
  if (index >= n_intervals) return;
  int interval_start = interval_starts[index];
  int interval_length = interval_lengths[index];
  float psum = 0;
  const float* cur_depth;
  const float* cur_feat;
  for(int i = 0; i < interval_length; i++){
    cur_depth = depth + ranks_depth[interval_start+i];
    cur_feat = feat + ranks_feat[interval_start+i] * c + cur_c;
    psum += *cur_feat * *cur_depth;
  }

  const int* cur_rank = ranks_bev + interval_start;
  float* cur_out = out + *cur_rank * c + cur_c; // cur_out is a pointer for out (+ offset), so change *cur_out will change the result of out
  *cur_out = psum;
}

// bev pool v2 前向 kernel 调用函数
void bev_pool_v2(int c, int n_intervals, const float* depth, const float* feat, const int* ranks_depth,
  const int* ranks_feat, const int* ranks_bev, const int* interval_starts, const int* interval_lengths, float* out) {
  bev_pool_v2_kernel<<<(int)ceil(((double)n_intervals * c / 256)), 256>>>(
    c, n_intervals, depth, feat, ranks_depth, ranks_feat,
    ranks_bev, interval_starts, interval_lengths, out
  );
}
```

确定 kernel 实现代码后需将代码拷贝到各 paddle 套件指定位置，自定义算子的 kernel 实现通常在各套件 ops 文件夹下，如 Paddle3D 的算子 kernel 实现位于```paddle3d/op/[op_name]```下

算子前向计算函数需遵循 Paddle 自定义算子规范，其中

  - 3.2. 函数输入类型确定

   函数输入类型只能是```paddle::Tensor```，``` std::vector<paddle::Tensor>``` 或``` Attribute```，其中```paddle::Tensor```需要以 const 的形式输入，```Attribute```只支持特定的数据类型，详细请参考[运算函数与基础 API](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/custom_op/new_cpp_op_cn.html#api)

bev pool v2 算子前向计算函数输入参数返回值如下：

```c++
std::vector<paddle::Tensor> bev_pool_v2_forward(
  const paddle::Tensor &_depth,
  const paddle::Tensor &_feat,
  const paddle::Tensor &_ranks_depth,
  const paddle::Tensor &_ranks_feat,
  const paddle::Tensor &_ranks_bev,
  const paddle::Tensor &_interval_lengths,
  const paddle::Tensor &_interval_starts,
  const std::vector<int> &_bev_feat_shape
)
```

  - 3.3. 函数的逻辑实现可参考 torch 对应的实现，通常会包含对 kernel 算子的调用
  - 3.4. 函数返回值只能是```std::vector<paddle::Tensor>``` 类型

完整的 bev pool v2 前向计算函数如下

```c++
  std::vector<paddle::Tensor> bev_pool_v2_forward(
    const paddle::Tensor &_depth,
    const paddle::Tensor &_feat,
    const paddle::Tensor &_ranks_depth,
    const paddle::Tensor &_ranks_feat,
    const paddle::Tensor &_ranks_bev,
    const paddle::Tensor &_interval_lengths,
    const paddle::Tensor &_interval_starts,
    const std::vector<int> &_bev_feat_shape
  ) {
    int c = _feat.shape()[4];
    int n_intervals = _interval_lengths.shape()[0];
    const float* depth = _depth.data<float>();
    const float* feat = _feat.data<float>();
    const int* ranks_depth = _ranks_depth.data<int>();
    const int* ranks_feat = _ranks_feat.data<int>();
    const int* ranks_bev = _ranks_bev.data<int>();

    const int* interval_lengths = _interval_lengths.data<int>();
    const int* interval_starts = _interval_starts.data<int>();

    auto _out = paddle::full(_bev_feat_shape, 0,
                              _feat.type(), paddle::GPUPlace()); //add to return out

    float* out = _out.data<float>();
    bev_pool_v2(
      c, n_intervals, depth, feat, ranks_depth, ranks_feat,
      ranks_bev, interval_starts, interval_lengths, out
    );
    return {_out};
  }
```

__4. 实现算子反向计算函数__

反向计算函数函数名以_backward 结束，其他的定义规范与前向函数相同

bev pool v2 反向计算函数实现如下

```c++
std::vector<paddle::Tensor> bev_pool_v2_backward(
  const paddle::Tensor &_out_grad,
  const paddle::Tensor &_depth,
  const paddle::Tensor &_feat,
  const paddle::Tensor &_ranks_depth,
  const paddle::Tensor &_ranks_feat,
  const paddle::Tensor &_ranks_bev,
  const paddle::Tensor &_interval_lengths,
  const paddle::Tensor &_interval_starts
) {
  int c = _out_grad.shape()[4];
  int n_intervals = _interval_lengths.shape()[0];
  const float* out_grad = _out_grad.data<float>();
  const float* depth = _depth.data<float>();
  const float* feat = _feat.data<float>();
  const int* ranks_depth = _ranks_depth.data<int>();
  const int* ranks_feat = _ranks_feat.data<int>();
  const int* ranks_bev = _ranks_bev.data<int>();
  const int* interval_lengths = _interval_lengths.data<int>();
  const int* interval_starts = _interval_starts.data<int>();

  int b = _feat.shape()[0];
  int h = _feat.shape()[2];
  int w = _feat.shape()[3];
  int d = _depth.shape()[2];
  int n = _depth.shape()[1];

  auto _depth_grad = paddle::full({n, d, h, w}, 0.0,
                          _depth.type(), paddle::GPUPlace());

  auto _feat_grad = paddle::full({n, h, w, c}, 0.0,
                          _feat.type(), paddle::GPUPlace());

  float* depth_grad = _depth_grad.data<float>();
  float* feat_grad = _feat_grad.data<float>();
  bev_pool_v2_grad(
    c, n_intervals, out_grad, depth, feat, ranks_depth, ranks_feat,
    ranks_bev, interval_starts, interval_lengths, depth_grad, feat_grad
  );
  return {{_depth_grad, _feat_grad}};
}
```

__5. 实现 shape 推导函数__

- 5.1 shape 推导函数输入类型为```std::vector<int64_t>```, 且输入参数需要包含所有计算函数函数中输入参数的 shape

- 5.2 返回类型 ```std::vector<int64_t>```， 表示前向计算输出 tensor 的维度

bev pool v2 shape 推导对应实现如下

```c++
std::vector<std::vector<int64_t>> BevPoolV2InferShape(
  std::vector<int64_t> _depth_shape,
  std::vector<int64_t> _feat_shape,
  std::vector<int64_t> _ranks_depth_shape,
  std::vector<int64_t> _ranks_feat_shape,
  std::vector<int64_t> _ranks_bev_shape,
  std::vector<int64_t> _interval_lengths_shape,
  std::vector<int64_t> _interval_starts_shape,
  const std::vector<int> _bev_feat_shape) {
    return {{_bev_feat_shape[0], _bev_feat_shape[1], _bev_feat_shape[2], _bev_feat_shape[3], _bev_feat_shape[4]}};
}
```

__6. 实现 dtype 推导函数__

- 6.1 shape 推导函数输入类型为```paddle::DataType```, 且输入参数需要包含所有计算函数函数中输入参数的 dtype
- 6.2 返回类型 ```std::vector<paddle::DataType>```， 表示前向计算输出 tensor 的 dtype

bev pool v2 dtype 推导对应实现如下

```c++
std::vector<paddle::DataType> BevPoolV2InferDtype(
  paddle::DataType _depth_dtype,
  paddle::DataType _feat_dtype,
  paddle::DataType _ranks_depth_dtype,
  paddle::DataType _ranks_feat_dtype,
  paddle::DataType _ranks_bev_dtype,
  paddle::DataType _interval_lengths_dtype,
  paddle::DataType _interval_starts_dtype) {
    return {_feat_dtype};
}
```

__7. 注册前向 op__

- 7.1 调用```PD_BUILD_OP(op_name)```
- 7.2 声明输入参数，调用```Inputs()```
- 7.3 声明```Attributes```（可选）, 调用```Attrs()```
- 7.4 声明输出，调用```Outputs()```
- 7.5 定义前向计算函数，调用```SetKernel()```
- 7.6 调用 shape 推导函数, 调用```SetInferShapeFn()```
- 7.7 调用 dtype 推导函数， 调用```SetInferDtypeFn()```

bev pool v2 注册前向 op 实现如下

```c++
PD_BUILD_OP(bev_pool_v2)
    .Inputs({"_depth", "_feat", "_ranks_depth",
            "_ranks_feat", "_ranks_bev", "_interval_lengths",
            "_interval_starts"})
    .Attrs({"_bev_feat_shape: std::vector<float>"})
    .Outputs({"out"})
    .SetKernelFn(PD_KERNEL(bev_pool_v2_forward))
    .SetInferShapeFn(PD_INFER_SHAPE(BevPoolV2InferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(BevPoolV2InferDtype));
```

__8. 注册反向 op__

- 8.1 调用```PD_BUILD_GRAD_OP(op_name)```
- 8.2 声明输入参数，调用```Inputs()```
- 8.3 申明```Attributes```(可选), 调用```Attrs()```
- 8.4 申明输出，调用```Outputs()```
- 8.5 定义前向计算函数，调用 ```SetKernelFn()```

bev pool v2 注册反向 op 实现如下

```c++
PD_BUILD_GRAD_OP(bev_pool_v2)
    .Inputs({paddle::Grad("_out"), "_depth", "_feat",
            "_ranks_depth",  "_ranks_feat", "_ranks_bev",
            "_interval_lengths", "_interval_starts", "_bev_feat_shape"})
     .Outputs({paddle::Grad("depth_grad"), paddle::Grad("feat_grad")})
     .SetKernelFn(PD_KERNEL(bev_pool_v2_backward));
```

#### 算子接入对应套件

算子定义实现完成后需要将算子接入对应的套件，以 bev_pool_v2 接入 Paddle3D 为例

在```paddle3d/ops/__init__.py``` 的 custom_op 列表中增加算子名和实现路径的映射即可

```python
custom_ops = {
    # ...
      # bev pool v2 算子名和实现路径的映射
    'bev_pool_v2': {
        'sources': [
            'bev_pool_v2/bev_pool.cc',
            'bev_pool_v2/bev_pool_cuda.cu'
        ],
        'version':
        '0.1.0',
    },
      # ...
  }
```

调用该自定义算子时只需要在其他文件中 import 该算子名即可，调用该算子的前向时只需要调用 PD_BUILD_OP 注册的前向函数名

如在 Paddle3D 中导入 bev pool v2 并调用其前向:

```python
from paddle3d.ops import bev_pool_v2

# 调用 bev pool v2 前向
bev_pool_v2.bev_pool_v2(
            depth,
            feat,
            ranks_depth,
            ranks_feat,
            ranks_bev,
            interval_lengths,
            interval_starts,
            bev_feat_shape
)
```

### 自定义 python 算子

__1. 继承```paddle.autograd.PyLayer```__

以 bevdet 中调用 bev pool v2 时定义的```QuickCumsumCuda```算子为例

```python
from paddle.autograd import PyLayer
class QuickCumsumCuda(PyLayer):

    # 算子实现部分
    # ...
```

__2. 实现中需包含```forward```函数和```backward```函数，均为```staticmethod```__

```python
from paddle.autograd import PyLayer
class QuickCumsumCuda(PyLayer):
    @staticmethod
    def forward(ctx, depth, feat, ranks_depth, ranks_feat, ranks_bev,
                bev_feat_shape, interval_starts, interval_lengths):
        # forward 实现

    @staticmethod
    def backward(ctx, out_grad):
        # backward 实现

```

__3. forward 中实现算子的前向逻辑__
前向逻辑实现中可包含对自定义 cuda 算子的调用，如需要保存 feature 结果在反向时使用，则需要调用```ctx.save_for_backward()```将保存 feature 值传入

```python
    @staticmethod
    def forward(ctx, depth, feat, ranks_depth, ranks_feat, ranks_bev,
                bev_feat_shape, interval_starts, interval_lengths):
        # 前向逻辑实现
        # ...
        #保存 feature 结果在 backward 时调用
        ctx.save_for_backward(ranks_bev, depth, feat, ranks_feat, ranks_depth)
        return out
```

__4. backward 中实现算子的反向逻辑__
反向逻辑实现中可包含对自定义 cuda 算子的调用，如需要用到 forward 中保存的 feature ，可以从```ctx.saved_tensor()```返回值拿到

```python
    @staticmethod
    def backward(ctx, out_grad):
        # 拿到 forward 中保存的 feature 结果
        ranks_bev, depth, feat, ranks_feat, ranks_depth = ctx.saved_tensor()
        # 反向逻辑实现
        # ...
        return depth_grad, feat_grad, None, None, None, None
```

__5. 通过 PyLayer 类的类方法 apply()调用 python 自定义算子__

```python
def bev_pool_v2_pyop(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                     bev_feat_shape, interval_starts, interval_lengths):
    x = QuickCumsumCuda.apply(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                              bev_feat_shape, interval_starts, interval_lengths)
    # ...
    return x
```
