# 自定义外部算子（旧）

> WARNING：这种自定义Op的方式在2.1版本将被废弃，推荐使用新自定义Op方案[《自定义外部算子（新）》](./new_custom_op.html)

通常，如果PaddlePaddle的Operator(OP)库中没有您所需要的操作，建议先尝试使用已有的OP组合，如果无法组合出您需要的操作，可以尝试使用`paddle.static.py_func`，也可以按照这篇教程自定义C++ OP。当然，如果用若干OP组合出来的OP性能无法满足您的要求，也可以自定义C++ OP。

自定义OP需要以下几个步骤:

1. 实现OP和注册OP，和在框架内部写OP完全相同，遵守"如何写新的C++ OP"的规范和步骤。当然，实现Gradient OP是可选的。
2. 编译出动态库。
3. 封装该OP的Python接口。
4. 写OP的单测。



下面通过一个具体的例子来详细的介绍，一步一步教会您如何实现。下面通过实现relu op来介绍。



##  自定义OP的实现

OP的实现与"如何写新的C++ OP"的教程相同，简答的说需要: 1). 定义OP的ProtoMaker，即描述OP的输入、输出、属性信息；2). 实现OP的定义和InferShape，以及OP的kernel函数，反向OP类似。3). 注册OP，以及OP的计算函数。

ReLU OP的CPU实现， ``relu_op.cc`` 文件:

```
// relu_op.cc
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

// 前向OP的输入X、输出Y、属性
class Relu2OpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input tensor.");
    AddOutput("Y", "Output of relu_op");
    AddComment(R"DOC(
Relu Operator.
Y = max(X, 0)
)DOC");
  }
};

// 前向OP的定义和InferShape实现，设置输出Y的shape
class Relu2Op : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    auto in_dims = ctx->GetInputDim("X");
    ctx->SetOutputDim("Y", in_dims);
  }
};

// 实现前向OP的Kernel计算函数: Y = max(0, X)
using Tensor = framework::Tensor;
template <typename DeviceContext, typename T>
class Relu2Kernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in_t = ctx.Input<Tensor>("X");
    auto* out_t = ctx.Output<Tensor>("Y");
    auto x = in_t->data<T>();
    // mutable_data分配内存、获取指针
    auto y = out_t->mutable_data<T>(ctx.GetPlace());
    for (int i = 0; i < in_t->numel(); ++i) {
      y[i] = std::max(static_cast<T>(0.), x[i]);
    }
  }
};

// 定义反向OP的输入Y和dY、输出dX、属性:
template <typename T>
class Relu2GradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  void Apply(GradOpPtr<T> op) const override {
    op->SetType("relu2_grad");
    op->SetInput("Y", this->Output("Y"));
    op->SetInput(framework::GradVarName("Y"), this->OutputGrad("Y"));
    op->SetAttrMap(this->Attrs());
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
  }
};

// 定义反向OP和InferShape实现,设置dX的shape
class Relu2GradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    auto in_dims = ctx->GetInputDim(framework::GradVarName("Y"));
    ctx->SetOutputDim(framework::GradVarName("X"), in_dims);
  }
};

// 实现反向OP的kernel函数 dx = dy * ( y > 0. ? 1. : 0)
template <typename DeviceContext, typename T>
class Relu2GradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* dy_t = ctx.Input<Tensor>(framework::GradVarName("Y"));
    auto* y_t = ctx.Input<Tensor>("Y");
    auto* dx_t = ctx.Output<Tensor>(framework::GradVarName("X"));

    auto dy = dy_t->data<T>();
    auto y = y_t->data<T>();
    auto dx = dx_t->mutable_data<T>(ctx.GetPlace());

    for (int i = 0; i < y_t->numel(); ++i) {
      dx[i] = dy[i] * (y[i] > static_cast<T>(0) ? 1. : 0.);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using CPU = paddle::platform::CPUDeviceContext;
// 注册前向和反向op
// 为了和框架内部的relu区分，这里注册的OP type为relu2
REGISTER_OPERATOR(relu2,
                  ops::Relu2Op,
                  ops::Relu2OpMaker,
                  ops::Relu2GradMaker<paddle::framework::OpDesc>,
                  ops::Relu2GradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(relu2_grad, ops::Relu2GradOp);
// 注册CPU的Kernel
REGISTER_OP_CPU_KERNEL(relu2,
                       ops::Relu2Kernel<CPU, float>,
                       ops::Relu2Kernel<CPU, double>);
REGISTER_OP_CPU_KERNEL(relu2_grad,
                       ops::Relu2GradKernel<CPU, float>,
                       ops::Relu2GradKernel<CPU, double>);
```



ReLU OP的GPU实现， ``relu_op.cu`` 文件:

```
// relu_op.cu
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
__global__ void KeRelu2(const T* x, const int num, T* y) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = gid; i < num; i += blockDim.x * gridDim.x) {
    y[i] = max(x[i], static_cast<T>(0.));
  }
}

// 前向OP的kernel的GPU实现
template <typename DeviceContext, typename T>
class Relu2CUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in_t = ctx.Input<Tensor>("X");
    auto* out_t = ctx.Output<Tensor>("Y");
    auto x = in_t->data<T>();
    auto y = out_t->mutable_data<T>(ctx.GetPlace());

    auto& dev_ctx = ctx.template device_context<DeviceContext>();

    int num = in_t->numel();
    int block = 512;
    int grid = (num + block - 1) / block;
    KeRelu2<T><<<grid, block, 0, dev_ctx.stream()>>>(x, num, y);
  }
};

template <typename T>
__global__ void KeRelu2Grad(const T* y, const T* dy, const int num, T* dx) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = gid; i < num; i += blockDim.x * gridDim.x) {
    dx[i] = dy[i] * (y[i] > 0 ? 1. : 0.);
  }
}

// 反向OP的kernel的GPU实现
template <typename DeviceContext, typename T>
class Relu2GradCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* dy_t = ctx.Input<Tensor>(framework::GradVarName("Y"));
    auto* y_t = ctx.Input<Tensor>("Y");
    auto* dx_t = ctx.Output<Tensor>(framework::GradVarName("X"));

    auto dy = dy_t->data<T>();
    auto y = y_t->data<T>();
    auto dx = dx_t->mutable_data<T>(ctx.GetPlace());

    auto& dev_ctx = ctx.template device_context<DeviceContext>();

    int num = dy_t->numel();
    int block = 512;
    int grid = (num + block - 1) / block;
    KeRelu2Grad<T><<<grid, block, 0, dev_ctx.stream()>>>(y, dy, num, dx);
  }
};

}  // namespace operators
}  // namespace paddle

using CUDA = paddle::platform::CUDADeviceContext;
// 注册前向的GPU Kernel
REGISTER_OP_CUDA_KERNEL(relu2,
                        paddle::operators::Relu2CUDAKernel<CUDA, float>,
                        paddle::operators::Relu2CUDAKernel<CUDA, double>);
// 注册反向的GPU Kernel
REGISTER_OP_CUDA_KERNEL(relu2_grad,
                        paddle::operators::Relu2GradCUDAKernel<CUDA, float>,
                        paddle::operators::Relu2GradCUDAKernel<CUDA, double>);
```

注意点:

1. OP的type不能和PaddlePaddle已有的OP type相同，否则在Python中使用时会报错。



## 自定义OP的编译

需要将实现的C++、CUDA代码编译成动态库，下面通过g++/nvcc编译，当然您也可以写Makefile或者CMake。



编译需要include PaddlePaddle的相关头文件，如上面代码  `paddle/fluid/framework/op_registry.h` ，需要链接PaddlePaddle的lib库。 可通过下面命令获取到:

```
# python
>>> import paddle
>>> print(paddle.sysconfig.get_include())
/paddle/pyenv/local/lib/python2.7/site-packages/paddle/include
>>> print(paddle.sysconfig.get_lib())
/paddle/pyenv/local/lib/python2.7/site-packages/paddle/libs
```

下面命令可编译出动态库:

```
include_dir=$( python -c 'import paddle; print(paddle.sysconfig.get_include())' )
lib_dir=$( python -c 'import paddle; print(paddle.sysconfig.get_lib())' )

echo $include_dir
echo $lib_dir

# PaddlePaddel >=1.6.1, 仅需要include ${include_dir} 和 ${include_dir}/third_party
nvcc relu_op.cu -c -o relu_op.cu.o -ccbin cc -DPADDLE_WITH_CUDA -DEIGEN_USE_GPU -DPADDLE_USE_DSO -DPADDLE_WITH_MKLDNN -Xcompiler -fPIC -std=c++11 -Xcompiler -fPIC -w --expt-relaxed-constexpr -O3 -DNVCC \
    -I ${include_dir} \
    -I ${include_dir}/third_party \

g++ relu_op.cc relu_op.cu.o -o relu2_op.so -shared -fPIC -std=c++11 -O3 -DPADDLE_WITH_MKLDNN \
  -I ${include_dir} \
  -I ${include_dir}/third_party \
  -L /usr/local/cuda/lib64 \
  -L ${lib_dir} -lpaddle_framework -lcudart
```



注意点:

1. 通过NVCC编译CUDA源文件时，需要加编译选项 `-DPADDLE_WITH_CUDA -DEIGEN_USE_GPU -DPADDLE_USE_DSO`，在框架源码中会使用这些宏定义进行条件编译。用户自定义的C++ OP实现编译时，选项的开启状态需要和核心框架编译行为一致。如`EIGEN_USE_GPU`是使用Eigen数学库的GPU实现时需要增加的编译选项。
2. 如果飞桨安装包中不包含MKLDNN库，则需要去掉编译选项`-DPADDLE_WITH_MKLDNN`。核心框架源码中(比如tensor.h)有使用此宏定义进行条件编译，该选项是否打开同样需要和核心框架编译行为保持一致。默认的飞桨安装包中含有MKLDNN库。
3. 可多个OP编译到同一个动态库中。
4. 通过pip方式安装的PaddlePaddle由GCC 4.8编译得到，由于GCC 4.8和GCC 5以上**C++11 ABI不兼容**，您编写的自定义OP，需要通过GCC 4.8编译。若是GCC 5及以上的环境上使用自定义OP，推荐使用[Docker安装PaddlePaddle](https://www.paddlepaddle.org.cn/install/doc/docker)，使得编Paddle和编译自定义OP的GCC版本相同。



## 封装Python Layer接口

需要使用  `paddle.incubate.load_op_library`  接口调用加载动态库，使得PaddlePaddle的主进程中可以使用用户自定义的OP。

```
# custom_op.py
import paddle.incubate as incubate
# 调用load_op_library加载动态库
incubate.load_op_library('relu2_op.so')

from paddle.incubate import LayerHelper

def relu2(x, name=None):
    # relu2的type和在OP中定义的type相同
    helper = LayerHelper("relu2", **locals())
    # 创建输出Variable
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(type="relu2", inputs={"X": x}, outputs={"Y": out})
    return out
```

注意点:

1. 一个动态库只需使用`paddle.incubate.load_op_library`在`paddle` import之后加载一次即可。
2. Python接口的封装和PaddlePaddle框架内部的封装相同，更多的示例也可以阅读源码中 `python/paddle/fluid/layers/nn.py`的代码示例。

## 单测测试

 可以写个简单的Python程序测试计算的正确性:

 静态图模式
```
import numpy as np
import paddle
from custom_op import relu2

paddle.enable_static()
data = paddle.static.data(name='data', shape=[None, 32], dtype='float32')
relu = relu2(data)
use_gpu = True  # or False
paddle.set_device('gpu' if use_gpu else 'cpu')
exe = paddle.static.Executor()

x = np.random.uniform(-1, 1, [4, 32]).astype('float32')
out, = exe.run(feed={'data': x}, fetch_list=[relu])
np.allclose(out, np.maximum(x, 0.))
```

 动态图模式
```
import numpy as np
import paddle
from custom_op import relu2

use_gpu = True  # or False
paddle.set_device('gpu' if use_gpu else 'cpu')

x = np.random.uniform(-1, 1, [4, 32]).astype('float32')
t = paddle.to_tensor(x)
out = relu2(t)
np.allclose(out.numpy(), np.maximum(x, 0.))
```

接下来可以在模型中使用您自定义的OP了!

## 如何在C++预测库中使用

暂时不支持在C++预测库中使用，后续会补充在C++预测库中的使用示例。

## FAQ

1. Q: 如果出现类似错误: `relu2_op.so: cannot open shared object file: No such file or directory` 以及 `libpaddle_framework.so: cannot open shared object file: No such file or directory`。

   A: 需要将`relu2_op.so`所在路径以及`libpaddle_framework.so`路径(即`paddle.sysconfig.get_lib()`得到路径)设置到环境变量LD_LIBRARY_PATH中:

     ```
      # 假如relu2_op.so路径是：`paddle/test`，对于Linux环境设置:
      export LD_LIBRARY_PATH=paddle/test:$( python -c 'import paddle; print(paddle.sysconfig.get_lib())'):$LD_LIBRARY_PATH
     ```
