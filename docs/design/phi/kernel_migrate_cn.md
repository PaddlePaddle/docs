# OpKernel 迁移指南

## 一、迁移背景

由于 Paddle fluid 的算子范式存在诸多问题，我们设计实现了新的 PHI 算子范式，后续 PHI 算子范式将逐渐统一并替代原有 fluid 算子范式，诸多执行调度的适配将仅围绕 PHI 算子范式进行（例如，新动态图执行引擎仅在 PHI 算子范式上才能充分发挥其性能优势），fluid 算子范式将逐渐被废弃并移除。详细的问题背景及设计概要请参考 [《飞桨高可复用算子库 PHI 设计文档》](./design_cn.md)。

因此，我们正在逐渐将 Python 2.x API 体系中运算类 API 关联的 Op 迁移至 `paddle/phi` 目录，目前我们已经基本完成了必要算子的  InferShape 函数与 CPU&GPU OpKernel 的迁移工作，本文档旨在帮助剩余异构设备 XPU/MKLDNN/NPU/MLU 等 OpKernel 的迁移。

## 二、迁移内容

迁移工作是将原先在 `paddle/fluid/operators` 目录下实现的相应异构设备或者第三方库的 OpKernel 实现，改写为 PHI 形式的函数式 Kernel，并放置到 `paddle/phi/kernels` 目录或者外部 CustomDevice repo 的 kernels 目录。具体地，以 `log_softmax` Op 为例，对于不同设备后端来讲，迁移的对象及放置位置如下：

- XPU：`paddle/fluid/operators/log_softmax_op_xpu.cc` 修改并迁移至 `paddle/phi/kernels/xpu/log_softmax_kernel.cc & log_softmax_grad_kernel.cc`
- MKLDNN：`paddle/fluid/operators/mkldnn/log_softmax_mkldnn_op.cc` 修改并迁移至 `paddle/phi/kernels/onednn/log_softmax_kernel.cc`
- NPU：`paddle/fluid/operators/log_softmax_op_npu.cc` 修改并迁移至外部 CustomDevice repo `PaddleCustomDevice/backends/npu/kernels/log_softmax_kernel.cc`
- MLU：`paddle/fluid/operators/log_softmax_op_mlu.cc` 修改并迁移至外部 MLU 对应的插件式硬件适配 repo 中

## 三、迁移步骤

迁移异构 OpKernel 可以按照以下几个步骤进行：

1. 创建 kernel.cc 文件
2. 迁移并改写原先的 OpKernel 实现，完成注册
3. 编译并通过单元测试
4. 移除原 OpKernel 文件

迁移参考 PR：

- where_index xpu kernel 迁移 PR：[#40255](https://github.com/PaddlePaddle/Paddle/pull/40255)
- log_softmax mkldnn kernel 迁移 PR：[#43941](https://github.com/PaddlePaddle/Paddle/pull/43941)

### 3.1 创建 kernel.cc 文件

首先，根据当前迁移的 Kernel 所属的设备，确定迁移后 Kernel 文件所应该放置的子目录，例如 xpu kernel 迁移后放置到 `paddle/phi/kernels/xpu` 子目录，mkldnn kernel 迁移后放置到 `paddle/phi/kernels/onednn` 子目录。

然后，在相应子目录中创建 `xxx_kernel.cc` 和 `xxx_grad_kernel.cc` 文件，创建文件后在文件头部添加 License，include 必要的头文件，一般来讲至少需要 include 以下两个头文件（NPU 和 MLU 不需要添加 kernel.h 头文件）：

> 注意：在 phi/kernels 子目录下，前向和反向的 kernel 是分不同文件放置的，此处和迁移前不同

```c++
#include "paddle/phi/kernels/log_softmax_kernel.h"

#include "paddle/phi/core/kernel_registry.h"
```

一个是相应 kernel 的声明头文件，另一个是 kernel 注册所需的头文件，其他必要的头文件，按需添加即可，原则上不添加非必要的头文件。

最后，添加命名空间和所迁移 kernel 的函数定义，这里函数定义需要和 xxx_kernel.h 中的函数声明一致，直接从头文件中拷贝至 .cc 中即可：

```c++
namespace phi {

template <typename T, typename Context>
void LogSoftmaxKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      int axis,
                      DenseTensor* out) {
}

}  // namespace phi
```

> 注意：
> 1. Kernel 函数定义和 xxx_kernel.h 中的函数声明需要一致，如果不一致，则当前异构 Kernel 使用了非标准的参数输入，属于不合规的行为，需要先规范 Kernel 的实现
> 2. 如果在 phi/kernels 目录下找不到相应 Kernel 的函数定义，则暂时不需要迁移该 Kernel

### 3.2 迁移并改写原先 OpKernel 实现，完成注册

完成文件创建与函数定义创建后，即可将原先的 OpKernel 函数实现拷贝至相应的 Kernel 函数体中。然后按照如下步骤完善代码：

1. 移除 ExecutionContext 相关逻辑
2. 替换对象类型或函数
3. 迁移依赖的工具函数
4. 完成注册

#### 3.2.1 移除 ExecutionContext 相关逻辑

第一步，以 `log_softmax_mkldnn_op` 为例，先将原先 LogSoftmaxMKLDNNKernel 的实现拷贝过来，删除掉其中从 ExecutionContext 获取参数的代码。

```c++
template <typename T, typename Context>
void LogSoftmaxKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      int axis,
                      DenseTensor* out) {
  // auto& dev_ctx =
  //     ctx.template device_context<platform::MKLDNNDeviceContext>();  // delete this line
  const auto& mkldnn_engine = dev_ctx.GetEngine();

  // const Tensor* x = ctx.Input<Tensor>("X"); // delete this line
  // Tensor* out = ctx.Output<Tensor>("Out"); // delete this line

  // int axis = ctx.Attr<int>("axis"); // delete this line
  axis = axis >= 0 ? axis : x->dims().size() + axis;

  LogSoftmaxMKLDNNHandler<T> handler(mkldnn_engine, ctx.GetPlace(), x, axis);

  auto src_memory_p = handler.AcquireSrcMemory(x);
  auto dst_memory_p = handler.AcquireDstMemory(out);

  auto logsoftmax_p = handler.AcquireForwardPrimitive();

  auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();
  logsoftmax_p->execute(
      astream,
      {{DNNL_ARG_SRC, *src_memory_p}, {DNNL_ARG_DST, *dst_memory_p}});
  astream.wait();

  out->set_mem_desc(dst_memory_p->get_desc());
}
```

移除时，需要确认函数传入的参数命名，和函数内实际使用的参数命名是一致的，如果不一致，需要将函数实现中的参数名替换为函数参数列表中的命名。在当前示例中，参数名刚好是一致的。

#### 3.2.2 替换对象类型或函数

第二步，需要对其他的函数实现写法进行更新。

首先，需要根据参数类型的变化，更新参数成员访问的方式，最主要的差别是，原先通过 ExecutionContext 取出的输入 Tensor 一般是指针，例如：`const Tensor* x = ctx.Input<Tensor>("X");`，但是迁移之后，输入 Tensor 统一改为了引用的写法，例如：`const DenseTensor& x`，因此参数的访问方式会有所变化。继续以 `log_softmax_mkldnn_op` 为例，这里需要改动的地方如下方注释所示：

```c++
template <typename T, typename Context>
void LogSoftmaxKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      int axis,
                      DenseTensor* out) {
  const auto& mkldnn_engine = dev_ctx.GetEngine();
  // axis = axis >= 0 ? axis : x->dims().size() + axis;
  axis = axis >= 0 ? axis : x.dims().size() + axis;

  // LogSoftmaxMKLDNNHandler<T> handler(mkldnn_engine, ctx.GetPlace(), x, axis);
  LogSoftmaxMKLDNNHandler<T> handler(mkldnn_engine, ctx.GetPlace(), &x, axis);

  // auto src_memory_p = handler.AcquireSrcMemory(x);
  auto src_memory_p = handler.AcquireSrcMemory(&x);
  auto dst_memory_p = handler.AcquireDstMemory(out);

  auto logsoftmax_p = handler.AcquireForwardPrimitive();

  auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();
  logsoftmax_p->execute(
      astream,
      {{DNNL_ARG_SRC, *src_memory_p}, {DNNL_ARG_DST, *dst_memory_p}});
  astream.wait();

  out->set_mem_desc(dst_memory_p->get_desc());
}
```

其次，需要将原先函数实现中部分仅在 fluid 中使用的类型或函数，替换为 PHI 中对应的类型或函数，一些替换的映射关系如下：

| fluid 写法 | phi 写法 |
|---|---|
| `framework::Tensor` | `DenseTensor` |
| `framework::DenseTensor` | `DenseTensor` |
| 模板参数 `DeviceContext` | 模板参数 `Context` |
| `platform::XXXDeviceContext` | `XXXContext` |
| `out->mutbale_data(ctx.GetPlace()/place)` | `dev_ctx.template Alloc(out)` |
| `auto* ptr = out->mutbale_data()` | `auto* ptr = out->data()` |
| `out->mutbale_data(dims, place)` | `out->Resize(dims); dev_ctx.template Alloc(out)` |
| `out->mutbale_data(place, dtype)` | `dev_ctx.Alloc(out, dtype)` |
| `platform::erros::XXX` | `phi::erros::XXX` |
| `platform::float16/bfloat16/complex64/complex128` | `dtype::float16/bfloat16/complex64/complex128` |
| `framework::Eigen***` | `Eigen***` |
| `platform::XXXPlace` | `phi::XXXPlace` |
| `framework::DefaultCPUGenerator()` | `dev_ctx.GetGenerator()->GetCPUEngine()` |
| `framework::LoD` | `phi::LoD` |
| `framework::TensorCopy/TensorCopySync` | `phi::Copy` |
| `platform::is_xxx_place` | `place.GetType() == phi::AllocationType::XXX` |

> 注意：PHI 最终会作为独立的库进行编译，服务于 fluid、infrt、自定义算子等上层组件，因此原则上 PHI 中的文件不能 include fluid 中的头文件，迁移时注意尽可能不要 include 非必要的 fluid 头文件

继续以 `log_softmax_mkldnn_op` 为例，这里需要将 `platform::MKLDNNDeviceContext` 替换为 `OneDNNContext` ，改动的地方如下方注释所示：

```c++
template <typename T, typename Context>
void LogSoftmaxKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      int axis,
                      DenseTensor* out) {
  const auto& mkldnn_engine = dev_ctx.GetEngine();
  axis = axis >= 0 ? axis : x.dims().size() + axis;

  LogSoftmaxMKLDNNHandler<T> handler(
      mkldnn_engine, dev_ctx.GetPlace(), x, axis);

  auto src_memory_p = handler.AcquireSrcMemory(&x);
  auto dst_memory_p = handler.AcquireDstMemory(out);

  auto logsoftmax_p = handler.AcquireForwardPrimitive();

  // auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();
  auto& astream = OneDNNContext::tls().get_stream();
  logsoftmax_p->execute(
      astream, {{DNNL_ARG_SRC, *src_memory_p}, {DNNL_ARG_DST, *dst_memory_p}});
  astream.wait();

  out->set_mem_desc(dst_memory_p->get_desc());
}
```

至此，Kernel 函数迁移与改写完成。

#### 3.2.3 迁移依赖的函数

第三步，迁移 Kernel 依赖的函数。

迁移 Kernel 时，Kernel 调用的相关 function 及 functor 也需要一并迁移到 phi，根据所依赖 function 或者 functor 使用场景不同，可以分为以下几种情况放置：

1. 仅有当前所迁移 Kernel 使用的辅助函数（具体到设备，比如 log_softmax 的 mkldnn Kernel），一律和 Kernel 实现放到同一个设备文件夹中
    - 如果辅助函数相关代码较少，就直接和 Kernel 实现放到同一个 .cc 文件中
    - 如果辅助函数相关代码较多，就在 Kernel 所在的设备目录创建相应的 xxx_utils.h 管理代码
2. 其他情况，在 phi/kernels/funcs 目录下创建 .h/cc 文件管理代码，如果当前依赖的辅助函数可以直接归类到 phi/kernels/funcs 目录下已有的文件中，则直接放过去，不必创建新的文件

继续以 `log_softmax_mkldnn_op` 为例，它的 Kernel 使用了 LogSoftmaxMKLDNNHandler，并且只有当前 Kernel 使用，符合上述情况 1，因此将其直接迁移到 log_softmax_kernel.cc 文件中，放到 Kernel 函数上方即可。

```c++
template <typename T>
class LogSoftmaxMKLDNNHandler
    : public paddle::platform::
          MKLDNNHandlerNoCachingT<T, dnnl::logsoftmax_forward> {
 public:
  LogSoftmaxMKLDNNHandler(const dnnl::engine mkldnn_engine,
                          Place cpu_place,
                          const DenseTensor& x,
                          const int axis)
      : paddle::platform::MKLDNNHandlerNoCachingT<T, dnnl::logsoftmax_forward>(
            mkldnn_engine, cpu_place) {
    this->AcquireForwardPrimitiveDescriptor(
        dnnl::prop_kind::forward_inference, x.mem_desc(), axis);
  }
};

template <typename T, typename Context>
void LogSoftmaxKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      int axis,
                      DenseTensor* out) {
  const auto& mkldnn_engine = dev_ctx.GetEngine();
  axis = axis >= 0 ? axis : x.dims().size() + axis;

  LogSoftmaxMKLDNNHandler<T> handler(
      mkldnn_engine, dev_ctx.GetPlace(), x, axis);

  auto src_memory_p = handler.AcquireSrcMemory(&x);
  auto dst_memory_p = handler.AcquireDstMemory(out);

  auto logsoftmax_p = handler.AcquireForwardPrimitive();

  auto& astream = OneDNNContext::tls().get_stream();
  logsoftmax_p->execute(
      astream, {{DNNL_ARG_SRC, *src_memory_p}, {DNNL_ARG_DST, *dst_memory_p}});
  astream.wait();

  out->set_mem_desc(dst_memory_p->get_desc());
}
```

> 注意：
> 1. 由于 MKLDNNHandlerNoCachingT 涉及代码较多，我们暂时未将其迁移至 phi 目录，因此这里仍然 include 了 fluid/platform 目录下的头文件，后续该部分代码也需要迁移至 phi 目录
> 2. 根据编程风格指南，输入参数一般是 const & 的形式，因此迁移 LogSoftmaxMKLDNNHandler 时将其输入参数 x 由指针改为 const & 形式，相应 LogSoftmaxKernel 内也改为直接传递 x 值的形式


#### 3.2.4 完成注册

第四步，添加 PHI Kernel 的注册代码。

注册 Kernel 的方式比较简单，在 paddle 内部直接使用 PD_REGISTER_KERNEL 宏注册即可。继续以 `log_softmax_mkldnn_op` 为例，注册写法如下：

```c++
PD_REGISTER_KERNEL(log_softmax,
                   OneDNN,
                   ALL_LAYOUT,
                   phi::LogSoftmaxKernel,
                   float,
                   phi::dtype::bfloat16) {}
```

字段说明：
1. log_softmax: Kernel 名称，和已有 CPU&GPU 设备注册的 Kernel 名称一致（注意不一定和原先 Op 的名称一致，需要参考 `paddle/phi/kernels/cpu/log_softmax_kernel.cc|cu` 文件末尾的注册的 Kernel 名称）
2. OneDNN: Backend 名称，目前在 paddle 内部迁移异构 Kernel 会遇到的包括 XPU，OneDNN
3. phi::LogSoftmaxKernel: Kernel 函数的名称，记得带上 namespace phi
4. 剩余的均为数据类型，注册的数据类型对齐原有 Kernel 即可。


> 注意
> 1. phi Kernel 的注册宏末尾是函数体  { }，不是直接加分号，此处与旧的注册宏有区别
> 2. 注册 Kernel 的宏声明需要在 global namespace
> 3. 少数算子迁移需要注意迁移到 PHI 下注册的 kernel 名称。比如 Fluid 下 reshape2 这个算子， Kernel 迁移到 PHI 下后注册更名为了  reshape ，这里存在一个 reshape2 到 reshape 的映射关系，通过`paddle/phi/ops/compat/reshape_sig.cc`文件里的 PD_REGISTER_BASE_KERNEL_NAME(reshape2, reshape)来体现。在迁移算子的过程中，如果未在 PHI 下找到对应名称的 CPU&GPU kernel ，可能是名字进行了映射，可以在 xxx_sig.cc 里找一下映射关系。
> 4. 在名字进行了映射的情况下，如果 Fluid 下存在和映射后 PHI 名字一样的算子，那么这个算子是我们废弃的算子，不需要迁移。比如 Fluid 下 reshape2 算子迁移到 PHI 下变成了 reshape ，那么 Fluid 下 reshape 算子就是废弃算子，不需要迁移。

对于在外部 CustomDevice 注册 Kernel，注册写法略有不同，以 log_softmax_op_npu(ascend) 为例，注册写法如下：

```c++
PD_REGISTER_PLUGIN_KERNEL(log_softmax,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::LogSoftmaxKernel,
                          float,
                          phi::dtype::float16,
                          double) {}
```

不同之处包括：
1. 注册宏的名称不一样，此处是 PD_REGISTER_PLUGIN_KERNEL
2. Backend 的名称是用户注册的 CustomDevice 的名称，此处是 ascend。

对于 Fluid 下通过宏 REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE 注册的 kernel ，迁移后按照正常 PHI 下所使用的宏注册就行，需要注意的是，如果 Fluid 下注册的 kernel 模板参数有俩个类型，由于 PHI 下注册的 Kernel 只支持一个类型， PHI 注册使用第一个类型。比如 Fluid 下的这个例子：

```c++

// Fluid Register

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(conv2d,
                                    MKLDNN,
                                    ::paddle::platform::CPUPlace,
                                    FP32,
                                    ops::kConvMKLDNNFP32,
                                    ops::ConvMKLDNNOpKernel<float, float>);

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(conv2d,
                                    MKLDNN,
                                    ::paddle::platform::CPUPlace,
                                    BF16,
                                    ops::kConvMKLDNNFP32,
                                    ops::ConvMKLDNNOpKernel<paddle::platform::bfloat16, float>);

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(conv2d,
                                    MKLDNN,
                                    ::paddle::platform::CPUPlace,
                                    U8,
                                    ops::kConvMKLDNNINT8,
                                    ops::ConvMKLDNNOpKernel<uint8_t, float>);

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(conv2d,
                                    MKLDNN,
                                    ::paddle::platform::CPUPlace,
                                    U8WS8,
                                    ops::kConvMKLDNNINT8WS8,
                                    ops::ConvMKLDNNOpKernel<uint8_t, int8_t>);

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(conv2d,
                                    MKLDNN,
                                    ::paddle::platform::CPUPlace,
                                    S8,
                                    ops::kConvMKLDNNINT8,
                                    ops::ConvMKLDNNOpKernel<int8_t, float>);

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(conv2d,
                                    MKLDNN,
                                    ::paddle::platform::CPUPlace,
                                    S8WS8,
                                    ops::kConvMKLDNNINT8WS8,
                                    ops::ConvMKLDNNOpKernel<int8_t, int8_t>);


// PHI Register
PD_REGISTER_KERNEL(conv2d,
                   OneDNN,
                   ALL_LAYOUT,
                   phi::ConvKernel,
                   float,
                   phi::dtype::bfloat16,
                   uint8_t,
                   int8_t) {}
```
这个例子中，原 Fluid 中对 Kernel 模板参数中第二个类型参数的选择是在 kernel 选择中完成的。迁移到 PHI 后，我们 Kernel 只有一个类型模板参数，所以在选择了 Kernel 后，对第二个模板参数的选择需要根据 conv2d 的逻辑在 Kernel 里实现，伪代码如下：

```c++

template <typename T, typename K, typename Context>
void ConvImpl(const Context& dev_ctx,
            const DenseTensor& input,
            const DenseTensor& filter,
            ... 其他参数省略,
            DenseTensor* out) {
    return;
}

template <typename T, typename Context>
void ConvKernel(const Context& dev_ctx,
                const DenseTensor& input,
                const DenseTensor& filter,
                ... 其他参数省略,
                DenseTensor* out) {
    if (std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value) {
        if (filter.dtype() == DataType::INT8) {
            ConvImpl<T, int8_t>(dev_ctx, input, filter, ... , out);
        }
        else {
            ConvImpl<T, float>(dev_ctx, input, filter, ... , out);
        }
    }
    else {
        ConvImpl<T, float>(dev_ctx, input, filter, ... , out);
    }
}

```

### 3.3 编译并通过单元测试

至此，一个 Kernel 的主体代码迁移工作已经完成，接下来，需要先重新 cmake & make 确认下是否有语法错误或者其他编译错误，根据错误的提示进行问题的修复。

PHI Kernel 的优先级高于 fluid OpKernel ，因此编译通过后，可以先通过执行相应的单元测试，验证迁移后 Kernel 的正确性，继续以 `log_softmax_mkldnn_op` 为例，这里可以验证 test_log_softmax_mkldnn_op.py 是否可以执行通过，通过意味着 Kernel 迁移正确。

`ctest -R test_log_softmax_mkldnn_op -V`


### 3.4 移除原 OpKernel 文件

前序步骤完成之后，需要移除原先 OpKernel 实现以及注册代码。一般来讲，可以直接删除原先 fluid operators 目录以下相应文件，继续以 `log_softmax_mkldnn_op` 为例，需要移除 `paddle/fluid/operators/mkldnn/log_softmax_mkldnn_op.cc` ，不维护冗余的代码。

> 注意，如果出现找不到符号的报错，可能需要将部分 C++单测中的 `USE_OP(fluid_op_type)` 手动改为 `USE_OP_ITSELF(fluid_op_type)` ，以及将 `USE_OP_DEVICE_KERNEL(fluid_op_type, MKLDNN)` 改为 `PD_DECLARE_KERNEL(phi_kernel_name, OneDNN, ALL_LAYOUT)`。


## 四、注意事项

1. Kernel 注册时 Layout 字段目前不太准确，例如对于 OneDNN 来讲，它 Kernel 的 Layout 应该是 ONEDNN，但是目前仍然声明是 ALL_LAYOUT，后续我们也会扩展这里的机制
2. 迁移的时候，尽可能避免对原 Kernel 实现的逻辑改动，如果觉得它原来写得不好，想要优化，可以拆分 PR 进行（担心出现性能变化，CI 又发现不了，后续导致 CE 模型性能下降）
3. 按照编程风格指南，迁移后 kernel.cc 文件中对应的 kernel.h 需要在最前面 include
4. 注意迁移 Kernel 之后，DenseTensor*的输出参数，仍然是返回值的定位，所以在 kernel 内注意只能写该参数的成员值，而不能读它的值用作逻辑判断，可能会出错
5. 异构设备 Kernel 迁移时，一些写法不确定的地方可以参考已有的 CPU&GPU Kernel
