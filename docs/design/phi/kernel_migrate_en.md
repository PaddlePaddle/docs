# OpKernel Migration Guide

## 1. Migration background

Due to many defects in Paddle fluid's operator paradigm, we designed and implemented a new PHI operator paradigm. The subsequent PHI operator paradigm will gradually unify and replace the original fluid operator paradigm. Many execution scheduling adaptations will only focus on the PHI operator paradigm (For example, the new dynamic graph execution engine can only take full advantage of its performance on the PHI operator paradigm), and the fluid operator paradigm will gradually be deprecated and removed. For detailed background and design summary, please refer to ["Paddle HIgh reusability operator library (PHI) Design Document"](./design_en.md).

Therefore, we are gradually migrating the Ops associated with the operation API in the Python 2.x API module to the `paddle/phi` directory. At present, we have basically completed the migration of the InferShape function and the CPU&GPU OpKernel of the necessary operators. This document aims to helps the migration of remaining OpKernels for heterogeneous devices or third-party library such as XPU/MKLDNN/NPU/MLU.

## 2. Migration content

The migration work is to rewrite the OpKernel implementation of the corresponding heterogeneous device or third-party library originally implemented in the `paddle/fluid/operators` directory into a functional Kernel in the form of PHI Kernel, and place it in the `paddle/phi/kernels` directory or the kernels directory of the external CustomDevice repo. Specifically, taking `log_softmax` Op as an example, for different device backends, the objects to be migrated and their placement positions are as follows:

- XPU: `paddle/fluid/operators/log_softmax_op_xpu.cc` modified and migrated to `paddle/phi/kernels/xpu/log_softmax_kernel.cc & log_softmax_grad_kernel.cc`
- MKLDNN: `paddle/fluid/operators/mkldnn/log_softmax_mkldnn_op.cc` modified and migrated to `paddle/phi/kernels/onednn/log_softmax_kernel.cc`
- NPU: `paddle/fluid/operators/log_softmax_op_npu.cc` modified and migrated to external CustomDevice repo `PaddleCustomDevice/backends/npu/kernels/log_softmax_kernel.cc`
- MLU: `paddle/fluid/operators/log_softmax_op_mlu.cc` modified and migrated to the plug-in hardware adaptation repo corresponding to the external MLU

## 3. Migration steps

Migrating heterogeneous OpKernels can be done in the following steps:

1. Create the kernel.cc file
2. Migrate and rewrite the original OpKernel implementation to complete the registration
3. Compile and pass unit tests
4. Remove the original OpKernel file

Migration reference PR:

- where_index xpu kernel migration PR: [#40255](https://github.com/PaddlePaddle/Paddle/pull/40255)
- log_softmax mkldnn kernel migration PR: [#43941](https://github.com/PaddlePaddle/Paddle/pull/43941)

### 3.1 Create the kernel.cc file

First, according to the device to which the currently migrated Kernel belongs, determine the subdirectory where the migrated Kernel files should be placed. For example, the xpu kernel is placed in the `paddle/phi/kernels/xpu` subdirectory after the migration, and the mkldnn kernel is placed in the `paddle/phi/kernels/onednn` subdirectory after the migration.

Then, create the `xxx_kernel.cc` and `xxx_grad_kernel.cc` files in the corresponding subdirectories. After the files are created, add the License to the header of the file, and include the necessary header files. Generally, at least the following two header files need to be included:

> Note: In the phi/kernels subdirectory, the forward and gradient kernels are placed in different files, which are different from the fluid OpKernel paradigm.

```c++
#include "paddle/phi/kernels/log_softmax_kernel.h"

#include "paddle/phi/core/kernel_registry.h"
```

One is the declaration header file of the corresponding kernel, the other is the header file required for kernel registration, and other necessary header files can be added as needed. In principle, unnecessary header files are not added.

Finally, add the namespace and the function definition of the migrated kernel. The function definition here needs to be consistent with the function declaration in xxx_kernel.h, just copy it directly from the header file to .cc:

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

> Note:
> 1. Kernel function definitions and function declarations in xxx_kernel.h need to be consistent. If they are inconsistent, the current heterogeneous Kernel uses non-standard parameter input, which is an illegal behavior, the implementation of Kernel needs to be standardized first.
> 2. If the function definition of the corresponding Kernel cannot be found in the phi/kernels directory, it is not necessary to migrate the Kernel temporarily.

### 3.2 Migrate and rewrite the original OpKernel implementation and complete the registration

After the file creation and function definition creation are completed, the original OpKernel function implementation can be copied to the corresponding PHI Kernel function body. Then follow the steps below to complete the rewrite:

1. Remove ExecutionContext related logic
2. Substitute object types or functions
3. Migrate tool functions that are depended on by Kernel
4. Complete the Kernel registration

#### 3.2.1 Remove ExecutionContext related logic

The first step, taking `log_softmax_mkldnn_op` as an example, copy the original implementation of LogSoftmaxMKLDNNKernel and delete the code that gets parameters from ExecutionContext.

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

When removing, you need to confirm that the parameter names passed in by the function are consistent with the parameter names actually used in the function. If they are inconsistent, you need to replace the parameter names in the function implementation with the names in the function parameter list. In the current example, the parameter names just coincide.

#### 3.2.2 Substitute object types or functions

The second step is to update other function implementations.

First of all, it is necessary to update the way of parameter member access according to the change of the parameter type. The main difference is that the input Tensor originally taken out through ExecutionContext is generally a pointer, for example: `const Tensor* x = ctx.Input<Tensor>("X ");`, but after the migration, the input Tensor is uniformly changed to const reference writing, for example: `const DenseTensor& x`, so the access method of parameters will change. Continuing to take `log_softmax_mkldnn_op` as an example, the changes that need to be made here are as follows:

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

Secondly, it is necessary to replace some of the types or functions that were only used in fluid in the original function implementation with the corresponding types or functions in PHI. The mapping relationship of some replacements are as follows:

| fluid writing | phi writing |
|---|---|
| `farmework::Tensor` | `DenseTensor` |
| `farmework::LoDTensor` | `DenseTensor` |
| template parameter `DeviceContext` | template parameter `Context` |
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

> Note: PHI will eventually be compiled as an independent library, serving upper-level components such as fluid, infrt, and custom operators. Therefore, in principle, the files in PHI cannot include the header files in fluid. When migrating, be careful not to include non-necessary fluid header files.

Continue to take `log_softmax_mkldnn_op` as an example, here you need to replace `platform::MKLDNNDeviceContext` with `OneDNNContext` , the changes are shown in the comments below:

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

So far, the Kernel function migration and rewriting is completed.


#### 3.2.3 Migrate tool functions that are depended on by Kernel

The third step is to migrate the functions that Kernel depends on.

When migrating the Kernel, the related functions and functors called by the Kernel also need to be migrated to phi together. According to the different usage scenarios of the functions or functors they depend on, they can be placed in the appropriate location according to the following situations:

1. Only the auxiliary functions used by the currently migrated Kernel (specific to the device, such as the mkldnn Kernel of log_softmax) are always placed in the same device folder as the Kernel implementation
    - If there is less code related to the auxiliary function, put it directly into the same .cc file as the Kernel implementation
    - If there are many auxiliary function related codes, create the corresponding xxx_utils.h management code in the device directory where the Kernel is located
2. In other cases, create the .h/cc file management code in the phi/kernels/funcs directory. If the currently dependent auxiliary functions can be directly classified into the existing files in the phi/kernels/funcs directory, put them directly there, without having to create a new file

Continue to take `log_softmax_mkldnn_op` as an example, its Kernel uses LogSoftmaxMKLDNNHandler, and only the current Kernel uses it, which conforms to the above situation 1, so it is directly migrated to the log_softmax_kernel.cc file and placed above the Kernel function.

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

> Note:
> 1. Since MKLDNNHandlerNoCachingT involves a lot of code, we have not migrated it to the phi directory for the time being, so the header files in the fluid/platform directory are still included here, and the subsequent part of the code also needs to be migrated to the phi directory.
> 2. According to the code style guide, the input parameters are generally in the form of `const &`, so when migrating LogSoftmaxMKLDNNHandler, change its input parameter x from a pointer to a `const &` form, and the corresponding LogSoftmaxKernel is also changed to the form of directly passing the value of x.

#### 3.2.4 Complete the Kernel registration

The fourth step is to add the registration code of the PHI Kernel.

We can directly use the `PD_REGISTER_KERNEL` macro to register inside the Paddle. Continue to take `log_softmax_mkldnn_op` as an example, the registration method is as follows:

```c++
PD_REGISTER_KERNEL(log_softmax,
                   OneDNN,
                   ALL_LAYOUT,
                   phi::LogSoftmaxKernel,
                   float,
                   phi::dtype::bfloat16) {}
```

Field Description:
1. log_softmax: Kernel name, which is the same as the Kernel name registered by the existing CPU&GPU device (note that it may not be the same as the original Op name, you need to refer to the registered Kernel name at the end of the `paddle/phi/kernels/cpu/log_softmax_kernel.cc|cu` file).
2. OneDNN: Backend name. Currently, when migrating heterogeneous Kernels inside paddle, XPU and OneDNN will be encountered.
3. phi::LogSoftmaxKernel: The name of the Kernel function, remember to bring namespace `phi`.
4. The rest are data types, and the registered data types can be aligned with the original Kernel.

> Note
> 1. The end of the registration macro of phi Kernel is the function body `{}`, instead of adding a `;` directly, which is different from the old registration macro.
> 2. The macro declaration of the registered Kernel needs to be in the global namespace.
> 3. A few operator migrations need to pay attention to the kernel name registered under the PHI. For example, the registered kernel `reshape2` in fluid is renamed to `reshape` when the kernel is migrated to PHI. There is a mapping relationship from `reshape2` to `reshape`. In file `paddle/phi/ops/compat/reshape_sig.cc`, the macro `PD_REGISTER_BASE_KERNEL_NAME(reshape2, reshape)` reflect this mapping relation. In the process of migrating operators, if the CPU & GPU kernel's name in Fluid is not found under the PHI, it may be that the name has been mapped. You can try to find the mapping relationship in `xxx_sig.cc`.
> 4. In the case of name mapping, if there is a kernel under the Fluid with the same name as the kernel(the name is mapped) in PHI, then this kernel is abandoned and does not need to be migrated. For example, the `reshape2` under fluid migrates to PHI and becomes `reshape`, so the `reshape` kernel under Fluid is an abandoned kernel and does not need to be migrated.

For registering Kernel in an external CustomDevice, the registration method is slightly different. Taking `log_softmax_op_npu(ascend)` as an example, the registration method is as follows:

```c++
PD_REGISTER_PLUGIN_KERNEL(log_softmax,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::LogSoftmaxKernel,
                          float,
                          phi::dtype::float16,
                          double) {}
```

Differences include:
1. The name of the registered macro is different, here is `PD_REGISTER_PLUGIN_KERNEL`.
2. The name of the Backend is the name of the CustomDevice registered by the user, here is `ascend`.

The kernel that registerd by REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE in Fluid is also registered using `PD_REGISTER_KERNEL` or `PD_REGISTER_PLUGIN_KERNEL` in PHI. It should be noted that if there are two types of kernel template parameters registered in fluid, since the kernel registered in PHI only supports one type, the first type is used in PHI registration. For example, the example under fluid:

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

In this example, in Fluid, the selection of the second template parameter in the kernel is completed in the kernel selection. After migrating to Phi, our kernel only has one template parameter for type, so after selecting the kernel, the selection of the second template parameter needs to be completed in the kernel according to the logic of conv2d. The pseudo code is as follows:


```c++

template <typename T, typename K, typename Context>
void ConvImpl(const Context& dev_ctx,
            const DenseTensor& input,
            const DenseTensor& filter,
            ... other params omit,
            DenseTensor* out) {
    return;
}

template <typename T, typename Context>
void ConvKernel(const Context& dev_ctx,
                const DenseTensor& input,
                const DenseTensor& filter,
                ... other params omit,
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

### 3.3 Compile and pass unit tests

So far, the main code migration of a Kernel has been completed. Next, you need to `cmake & make` to confirm whether there are syntax errors or other compilation errors, and repair the problem according to the error prompts.

The priority of PHI Kernel is higher than that of fluid OpKernel, so after the compilation is passed, you can first execute the corresponding unit test to verify the correctness of the Kernel after migration. Continue to take `log_softmax_mkldnn_op` as an example, here you can verify whether `test_log_softmax_mkldnn_op.py` can be execute pass, passing means that the Kernel is migrated correctly.

`ctest -R test_log_softmax_mkldnn_op -V`

### 3.4 Remove the original OpKernel file

After the previous steps are completed, the original OpKernel implementation and registration code need to be removed. Generally, you can directly delete the corresponding files under the original fluid operators directory, continue to take `log_softmax_mkldnn_op` as an example, you need to remove `paddle/fluid/operators/mkldnn/log_softmax_mkldnn_op.cc`, we do not maintain redundant code.

> Note: If there is an error that the symbol cannot be found, you may need to manually change `USE_OP(fluid_op_type)` in some C++ single tests to `USE_OP_ITSELF(fluid_op_type)`, and change `USE_OP_DEVICE_KERNEL(fluid_op_type, MKLDNN)` to ` PD_DECLARE_KERNEL(phi_kernel_name, OneDNN, ALL_LAYOUT)`.


## 4. Matters needing attention

1. Kernels registered through the macro `REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE` do not support migration to phi for the time being, and we will expand the mechanism support in the future.
2. The Layout field of Kernel registration is currently not accurate. For example, for OneDNN, its Kernel Layout should be ONEDNN, but it is still declared as ALL_LAYOUT. We will expand the mechanism here in the future.
3. When migrating, try to avoid logical changes to the original Kernel implementation. If you feel that it was originally poorly written and you want to optimize it, you can split the PR (Worried about performance changes, CI can't find it again, and the performance of CE model will be degraded later).
4. According to the code style guide, the corresponding kernel.h in the kernel.cc file after migration needs to be included at the top.
5. Note that after migrating the Kernel, the output parameter of DenseTensor* is still the location of the return value, so in the kernel, pay attention to only write the member value of this parameter, but not read its value for logical judgment, which may go wrong.
6. When migrating heterogeneous device Kernels, you can refer to the existing CPU&GPU Kernels for uncertain spellings.
