# 自定义Kernel举例

通过实现自定义后端CustomCPU（实际为CPU）的一个简单自定义kernel `abs` ，介绍其具体的使用流程。其中`abs`的功能可参考[Paddle官网API文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/abs_cn.html#abs)。


> 注意：
> - 在使用本机制实现自定义Kernel之前，请确保已经正确安装了[飞桨develop](https://github.com/PaddlePaddle/Paddle)最新版本
> - 当前仅支持 `Linux`平台
> - 仅支持Phi算子库已开放Kernel声明的自定义编码与注册


## 第一步：确定Kernel声明

查找飞桨发布的头文件`abs_kernel.h`中，其Kernel函数声明如下：

```c++
// Abs 内核函数
// 模板参数： T - 数据类型
//          Context - 设备上下文
// 参数： ctx - Context 对象
//       x - DenseTensor 对象
//       out - DenseTensor 指针
// 返回： None
template <typename T, typename Context>
void AbsKernel(const Context& ctx, const DenseTensor& x, DenseTensor* out);

```

## 第二步：Kernel实现与注册

```c++
// customcpu_abs_kernel.cc

#include "paddle/phi/extension.h" // 自定义Kernel依赖头文件

#include <cmath>  // CustomCPU封装库头文件，根据具体硬件封装库按需添加

namespace custom_cpu {

// Kernel函数体实现
template <typename T, typename Context>
void AbsKernel(const Context& ctx,
               const phi::DenseTensor& x,
               phi::DenseTensor* out) {
  // 使用dev_ctx的Alloc API为输出参数out分配模板参数T数据类型的内存空间
  dev_ctx.template Alloc<T>(out);
  // 使用DenseTensor的numel API获取Tensor元素数量
  auto numel = x.numel();
  // 使用DenseTensor的data API获取输入参数x的模板参数T类型的数据指针
  auto x_data = x.data<T>();
  // 使用DenseTensor的data API获取输出参数out的模板参数T类型的数据指针
  auto out_data = out->data<T>();
  // 完成计算逻辑
  for (auto i = 0; i < numel; ++i) {
    // 基于硬件相关封装库API abs完成逻辑计算
    out_data[i] = abs(x_data[i]);
  }
}

} // namespace custom_cpu

// 全局命名空间
// CustomCPU的AbsKernel注册
// 参数： abs - Kernel名称
//       CustomCPU - 后端名称
//       ALL_LAYOUT - 内存布局
//       custom_cpu::AbsKernel - Kernel函数名
//       float - 数据类型名
//       double - 数据类型名
//       phi::dtype::float16 - 数据类型名
PD_REGISTER_PLUGIN_KERNEL(abs,
                          CustomCPU,
                          ALL_LAYOUT,
                          custom_cpu::AbsKernel,
                          float,
                          double,
                          phi::dtype::float16){}
```

## 第三步：编译与使用

针对自定义硬件，自定义Kernel的使用与自定义Runtime绑定使用，且保持注册使用的backend参数与自定义Runtime中注册的名称一致。

TBD：链接到示例部分 或者 这部分直接放到整体举例中即可？

编译安装后，在Python安装路径paddle-plugins下新增了`libpaddle_custom_cpu.so`动态库，飞桨框架自动识别此路径进行动态库加载。

通过飞桨接口可见已注册`CustomCPU`的`Abs`Kernel，具体如下：

```bash
>>> paddle.fluid.core._get_all_register_op_kernels('phi')['abs']
...
`data_type[float]:data_layout[Undefined(AnyLayout)]:place[Place(CustomCPU:0)]:library_type[PLAIN]`
`data_type[::paddle::platform::float16]:data_layout[Undefined(AnyLayout)]:place[Place(CustomCPU:0)]:library_type[PLAIN]`
`data_type[double]:data_layout[Undefined(AnyLayout)]:place[Place(CustomCPU:0)]:library_type[PLAIN]`
```

选定自定义Runtime，并执行计算：

```python
>>> x = paddle.to_tensor([-1, -2, -3, -4], dtype='float32')
>>> x
Tensor(shape=[4], dtype=float32, place=Place(CustomCPU:0), stop_gradient=True,
       [-1., -2., -3., -4.])
>>> paddle.abs(x)
Tensor(shape=[4], dtype=float32, place=Place(CustomCPU:0), stop_gradient=True,
       [1., 2., 3., 4.])
```

> 注意：
> 1. 添加预定义宏`PADDLE_WITH_CUSTOM_DEVICE`以支持CustomContext
> 2. 添加预定义宏`PADDLE_WITH_CUSTOM_KERNEL`以隔离部分内部未开放的代码，未来会移除
> 3. 对于可能依赖的第三方库如`boost`，`glog`，`gflag`等，建议使用与飞桨框架依赖的相同版本
