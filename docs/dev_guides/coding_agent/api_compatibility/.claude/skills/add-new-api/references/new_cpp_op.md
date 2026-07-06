# 新增 C++ 算子

本文档介绍如何在飞桨框架中开发 C++ 算子，主要包括算子描述定义、Kernel 实现、Python API 封装、单元测试编写等内容。

## 一、获取并分析 PyTorch API 信息

若尚未获得 PyTorch API 的相关信息，则自行获取，获取方式请参考`api-compatibility/SKILL.md` 中的「API 信息获取方式」内容。


然后分析 PyTorch API 的功能和行为，在 Paddle 中新增对应的 C++ 算子和 Python API，使其与 PyTorch API 保持一致。具体包括：API 名称、调用路径、参数名及参数功能等。

## 二、开发流程概述

新增一个 C++ 算子需要以下步骤：
1. 新增算子描述及定义：描述前反向算子的输入、输出、属性，实现 InferMeta 函数
2. 新增算子 Kernel：实现算子在各种设备上的计算逻辑
3. 封装 Python API：封装 Python 端调用算子的接口
4. 添加单元测试：验证新增算子的正确性

各步骤对应的文件位置（假设算子名为 xxx）：
- 算子描述及定义：前向算子定义在 Paddle/paddle/phi/ops/yaml/ops.yaml，反向算子定义在 Paddle/paddle/phi/ops/yaml/backward.yaml
- 算子 InferMeta：Paddle/paddle/phi/infermeta/ 目录下的相应文件
- 算子 Kernel：Paddle/paddle/phi/kernels/ 目录下的 xxx_kernel.h、xxx_kernel.cc、xxx_grad_kernel.h、xxx_grad_kernel.cc 等文件
- Python API：Paddle/python/paddle/ 目录下的相应子目录中的 .py 文件
- 单元测试：Paddle/test/legacy_test/ 目录下的 test_xxx_op.py

用户使用飞桨开发神经网络模型时使用的 Python 接口（如 paddle.add()，paddle.relu()等）称为飞桨的 Python API，每个运算类的 Python API 在框架内部都会对应到一个或多个 C++ 端算子，每个算子在不同硬件设备上（CPU，GPU 等）实现的运算逻辑代码称为 Kernel。算子 InferMeta 函数是在算子 kernel 执行前将输出结果的维度、数据类型等信息进行处理，每个算子只需要实现一个 InferMeta 函数。

**Python API、算子 Yaml 配置、算子 InferMeta 函数和算子 Kernel 之间的关系：**

Python API 执行时会进入到 C++ 端由框架进行调度并执行相应的算子逻辑，算子的执行主要包括两个过程：
1. 执行算子 InferMeta 函数完成输出结果的维度、数据类型等静态信息的推导
2. 根据输入变量的设备信息选择对应的硬件设备来执行算子 Kernel，完成输出结果的数值计算

Python API 到算子 InferMeta 函数和 Kernel 调用之间的框架调度部分的逻辑代码主要通过算子 Yaml 配置中的信息自动生成。

接下来以 paddle.trace 为例，介绍如何新增算子。trace 算子用于计算输入 Tensor 在指定平面上的对角线元素之和，并输出相应的计算结果。

trace 示例代码路径：
- 算子描述及定义：Paddle/paddle/phi/ops/yaml/ops.yaml、Paddle/paddle/phi/ops/yaml/backward.yaml
- 算子 InferMeta：Paddle/paddle/phi/infermeta/unary.cc
- 算子 Kernel：Paddle/paddle/phi/kernels/trace_kernel.h、Paddle/paddle/phi/kernels/cpu/trace_kernel.cc、Paddle/paddle/phi/kernels/gpu/trace_kernel.cu、Paddle/paddle/phi/kernels/trace_grad_kernel.h、Paddle/paddle/phi/kernels/cpu/trace_grad_kernel.cc、Paddle/paddle/phi/kernels/gpu/trace_grad_kernel.cu
- Python API：Paddle/python/paddle/tensor/math.py
- 单元测试：Paddle/test/legacy_test/test_trace_op.py

## 三、新增算子描述及定义

算子描述及定义主要是定义算子的基本属性，包括算子的输入、输出以及各项非计算逻辑的配置，这些都是设备无关的。

### 3.1 算子 Yaml 文件配置

在 Paddle/paddle/phi/ops/yaml/ops.yaml 和 Paddle/paddle/phi/ops/yaml/backward.yaml 文件中对算子进行描述及定义，在框架编译时会根据 YAML 文件中的配置自动生成 C++ 端的相关代码接口以及内部实现。

Paddle/paddle/phi/ops/yaml/ops.yaml 中 trace 相关配置：

```yaml
- op : trace
  args : (Tensor x, int offset = 0, int axis1 = 0, int axis2 = 1)
  output : Tensor(out)
  infer_meta :
    func : TraceInferMeta
  kernel :
    func : trace
  backward : trace_grad
```

Paddle/paddle/phi/ops/yaml/backward.yaml 中 trace 相关配置：

```yaml
- backward_op : trace_grad
  forward : trace (Tensor x, int offset, int axis1, int axis2) -> Tensor(out)
  args : (Tensor x, Tensor out_grad, int offset, int axis1, int axis2)
  output : Tensor(x_grad)
  infer_meta :
    func : UnchangedInferMeta
    param : [x]
  kernel :
    func : trace_grad
    data_type : x
  no_need_buffer : x
```

ops.yaml 和 backward.yaml 分别对算子的前向和反向进行配置。

#### 前向算子基本配置项

- **op**：算子名称，与该算子 Python API 函数名相同（命名方式为：全小写+下划线），示例中为 trace
- **args**：算子输入参数，与该算子 Python API 函数的输入参数对应。当前支持的输入数据类型包括：Tensor, Tensor[], float, double, bool, int, int64_t, int[], int64_t[], str, Place, DataType, DataLayout, IntArray, Scalar。Tensor 类型的参数称为 Input（输入），非 Tensor 类型的参数称为 Attribute（属性）。注：Tensor[]表示 Tensor 数组；IntArray 为 int 类型数组，主要用于表示 shape,index 和 axes 等类型数据；Scalar 表示标量，可以支持不同的普通数据类型
- **output**：算子输出类型（目前支持 Tensor 和 Tensor[]类型），多个输出间用逗号","分隔开。可以使用"()"选择性标记输入的名字，如未标记默认为'out'。注：当返回类型为 Tensor[]时，需要在 Tensor[]后的'{}'内通过表达式指定返回数组的 size，如：Tensor[](out){input.size()}
- **infer_meta**：InferMeta 函数负责根据输入变量推断返回 Tensor 的维度与类型
- **infer_meta:func**：调用的 InferMeta 函数，示例中为 TraceInferMeta
- **infer_meta:param**：InferMeta 函数的输入参数，可以对 args 中的参数进行选择传入，未配置则默认传入 args 中的所有参数
- **kernel**：算子的计算 Kernel 配置
- **kernel:func**：算子对应 kernel 函数的注册名
- **kernel:param**：kernel 函数的输入参数，配置规则与 infer_meta:param 相同
- **kernel:data_type**：根据指定参数推导调用 kernel 的 data_type（对应 kernel 函数的模板参数'T'），默认不配置会根据输入 Tensor 自动推导。如果 kernel 的 data_type 类型由某个输入参数决定，需要将该参数的变量名填入该项
- **kernel:backend**：根据指定参数来选择调用 kernel 的 Backend（Kernel 执行的具体设备），默认不配置会根据输入 Tensor 自动推导
- **backward**：算子对应的反向算子名称，如果没有反向则不需要配置

#### 前向算子特殊配置项

- **optional**：指定输入 Tensor 为可选输入，用法可参考 dropout 中的 seed_tensor（位于 Paddle/paddle/phi/ops/yaml/ops.yaml）
- **inplace**：算子对指定的输入做原位处理并作为输出结果返回，使用格式：(x -> out)。特殊规则：如果 api 中算子名称有'_'后缀则只生成支持 inplace 功能的接口，如果算子名称没有'_'后缀，则会同时生成支持 inplace 操作的接口和不支持 inplace 的普通接口
- **view**：与 inplace 机制类似，区别在于 view 模式返回的结果只是与输入共享内存，使用格式：(x -> out)
- **intermediate**：标记前向计算中输出的用于反向计算的中间变量，不会出现在 Python API 的返回结果中，新增算子时不建议使用
- **invoke**：复用已有的算子接口或实现自定义的 C++ API，配置时以函数调用的形式配置即可，使用 invoke 时不需要配置 infer_meta 和 kernel
- **data_transform**：控制算子输入参数的自动转换行为，包括类型（dtype）、设备（backend）和布局（layout）
- **data_transform:skip_transform**：跳过指定参数的所有数据转换
- **data_transform:support_trans_dtype**：开启指定参数的自动类型转换

#### 反向算子配置项

- **backward_op**：反向算子名称，一般命名方式为：前向算子名称+'_grad'
- **forward**：对应前向算子的名称、参数、返回值，需要与 ops.yaml 中前向算子配置一致
- **args**：反向算子输入参数。约束：所有参数需要在 forward 配置项的参数中找到对应；反向输入参数需要按顺序排列：前向输入 Tensor、前向输出 Tensor、前向输出 Tensor 的反向梯度、前向非 Tensor 类型属性变量
- **output**：反向算子输出，顺序需要与前向输入 Tensor 一致
- **infer_meta** / **kernel** / **data_transform**：配置规则与前向算子相同
- **no_need_buffer**：标记的 Tensor 变量在前向运行完成后，持有的内存或显存会被释放。注意：由于 Tensor 内存被释放后会影响 dtype 接口的使用，所以需要在 kernel 的 data_type 配置项中指定其他的 Tensor 来推导 kernel 的 data_type

### 3.2 实现 InferMeta 函数

InferMeta 函数是根据输入参数，推断算子输出 Tensor 基本信息的函数，推断的信息包括输出 Tensor 的 shape、data type，同时它也承担了检查输入数据维度、类型等是否合法的功能。

InferMeta 与 kernel 共同组成了一个算子的运算过程。InferMeta 在 kernel 前执行，用于维度、数据类型等信息的计算处理，kernel 中不再需要专门推导这些信息。

trace 算子的 InferMeta 函数实现在 Paddle/paddle/phi/infermeta/unary.cc 中，主要逻辑包括：参数校验（维度>=2、axis 范围检查、axis1!=axis2）、计算输出维度。示例：

```cpp
void TraceInferMeta(const MetaTensor& x, int offset, int axis1, int axis2, MetaTensor* out) {
  auto x_dims = x.dims();
  PADDLE_ENFORCE_GE(x_dims.size(), 2,
                    errors::InvalidArgument("Input(X)'s rank is %d. Must be >= 2.", x_dims.size()));
  // ... 更多参数校验 ...
  auto sizes = common::vectorize(x_dims);
  if (x_dims.size() == 2) {
    sizes.clear();
    sizes.push_back(1);
  } else {
    // 移除 axis1 和 axis2 对应的维度
    int axis1_ = axis1 < 0 ? axis1 + x_dims.size() : axis1;
    int axis2_ = axis2 < 0 ? axis2 + x_dims.size() : axis2;
    sizes.erase(sizes.begin() + std::max(axis1_, axis2_));
    sizes.erase(sizes.begin() + std::min(axis1_, axis2_));
  }
  out->set_dims(common::make_ddim(sizes));
  out->set_dtype(x.dtype());
}
```

其中，MetaTensor 是对底层异构 Tensor 的抽象封装，仅支持对底层 Tensor 的维度、数据类型、布局等属性进行读取和设置，具体方法请参考 Paddle/paddle/phi/core/meta_tensor.h。

**InferMeta 的实现位置**（paddle/phi/infermeta/ 目录下，以 Tensor 输入个数为判定标准）：
- nullary.h：没有输入 Tensor 参数的函数
- unary.h：仅有一个输入 Tensor 参数的函数
- binary.h：有两个输入 Tensor 参数的函数
- ternary.h：有三个输入 Tensor 参数的函数
- multiary.h：有三个以上输入 Tensor 或者输入为 vector<Tensor> 的函数
- backward.h：反向算子的 InferMeta 函数一律在此文件中

**InferMeta 的编译时与运行时**

在静态图模型中，InferMeta 操作在编译时和运行时都会被调用。在 compile time 时，由于真实的维度未知，框架内部用 -1 来表示；在 run time 时，用实际的维度表示。因此维度的值在 compile time 和 run time 时可能不一致，如果存在维度的判断和运算操作，InferMeta 就需要区分 compile time 和 run time。

对于此类 InferMeta 函数，需要在 InferMeta 函数声明的参数列表末尾增加 MetaConfig 参数，例如：

```cpp
void ConcatInferMeta(const std::vector<MetaTensor*>& x,
                     const Scalar& axis_scalar,
                     MetaTensor* out,
                     MetaConfig config = MetaConfig());
```

然后在函数体中，使用 config.is_runtime 判断处于编译时还是运行时。

以下两种情况需要区分 compile time 和 run time：
- 检查：compile time 时不判断维度等于 -1 的情况，但在 runtime 时检查
- 运算：-1 和其他数做任何运算都要等于 -1

参考代码：
- 判断实现参考 Paddle/paddle/phi/infermeta/multiary.cc 中的 SigmoidCrossEntropyWithLogitsInferMeta 函数
- 运算实现参考 Paddle/paddle/phi/infermeta/multiary.cc 中的 ConcatInferMeta 函数

## 四、新增算子 Kernel

### 4.1 Kernels 目录结构

新增算子 Kernel 在 Paddle/paddle/phi/kernels/ 目录中完成，基本目录结构：
- 根目录：放置设备无关的 kernel 声明和实现
- cpu：仅放置 cpu 后端的 kernel 实现
- gpu：仅放置 gpu 后端的 kernel 实现
- xpu：仅放置百度 kunlun 后端的 kernel 实现
- funcs：放置一些支持多设备的、在多个 kernel 中使用的公共 functor 和 functions

新增算子仅需要关注 kernels 根目录及 kernel 所支持设备的子目录：
- kernels 根目录：放置设备无关的 kernel.h 和 kernel.cc。如果 kernel 除了一些简单的设备无关的 C++ 逻辑，关键计算逻辑均是复用已有的 kernel 函数实现的，那么它的声明和实现均直接放置到 kernels 目录下即可
- kernels 下一级子目录：放置特定后端的 kernel 实现代码

典型 kernel 新增时文件放置位置（假设算子名为 xxx）：
- 新增与设备无关的 kernel：新增文件包括 Paddle/paddle/phi/kernels/xxx_kernel.h、Paddle/paddle/phi/kernels/xxx_kernel.cc。反向 kernel 使用 grad_kernel 后缀
- 新增与设备相关、且 CPU & GPU 分别实现的 kernel：CPU 实现位于 Paddle/paddle/phi/kernels/cpu/ 目录下；GPU 实现位于 Paddle/paddle/phi/kernels/gpu/ 下。新增文件包括：Paddle/paddle/phi/kernels/xxx_kernel.h、Paddle/paddle/phi/kernels/cpu/xxx_kernel.cc、Paddle/paddle/phi/kernels/gpu/xxx_kernel.cu

### 4.2 Kernel 写法

#### 4.2.1 声明 Kernel 函数

以 trace 算子为例，首先在 Paddle/paddle/phi/kernels/ 目录下新建 Paddle/paddle/phi/kernels/trace_kernel.h 文件，用于放置前向 kernel 函数声明。

注意：
- Kernel 函数声明的参数列表原则上与 Python API 参数列表一致
- 所有的 kernel 声明，统一放在 namespace phi 中

```cpp
namespace phi {
template <typename T, typename Context>
void TraceKernel(const Context& ctx,
                 const DenseTensor& x,
                 int offset,
                 int axis1,
                 int axis2,
                 DenseTensor* out);
}
```

模板为固定写法：
- 第一个模板参数为数据类型 T，第二个模板参数为设备上下文 Context
- 函数命名：kernel 的命名统一加 Kernel 后缀，驼峰式命名
- 参数顺序：Context，InputTensor..., Attribute..., OutTensor*
- 第 1 个函数参数，类型为 const Context& 的 dev_ctx
- 第 2 个函数参数，输入 Tensor，类型一般为 const DenseTensor&
- 第 3-5 个函数参数，均为 attribute，多个 attribute 可以参考 Python 端 API 定义的顺序
- 第 6 个函数参数，输出 Tensor，类型一般为 DenseTensor*

特殊情况说明：
- 特殊模板参数：对于某些 kernel（如 reshape，copy），这些 kernel 不关注数据类型 T，可以省去第一个模板参数
- 特殊输入类型：对于某些特殊 kernel（如 concat 和 split kernel）的部分输入或输出是数组类型的 DenseTensor，此时输入类型为 const std::vector<const DenseTensor*>&；输出类型为 std::vector<DenseTensor*>

#### 4.2.2 实现 Kernel 函数

**复用已有 Kernel 实现设备无关 Kernel 函数**

以 linear 算子 (out = x * w + b) 为例介绍复用已有 kernel 实现设备无关 Kernel 函数的方法：

```cpp
#include "paddle/phi/kernels/elementwise_add_kernel.h"
#include "paddle/phi/kernels/elementwise_multiply_kernel.h"

template <typename T, typename Context>
void LinearKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  const DenseTensor& w,
                  const DenseTensor& b,
                  DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);         // 为 out 分配内存
  MultiplyKernel<T>(dev_ctx, x, w, out);  // 复用 MultiplyKernel
  AddKernel<T>(dev_ctx, out, b, out);     // 复用 AddKernel
}
```

复用 kernel 的流程：
1. 在源文件中 include 要复用 kernel 的头文件
2. 直接调用相应的 kernel 函数进行复用

注意：设备无关 kernel 实现时计算逻辑部分只能复用现有 kernel 或设备无关的 functor，不能使用设备相关的语法或者函数接口

**实现设备相关 Kernel 函数**

trace 算子的 CPU kernel 实现位于 Paddle/paddle/phi/kernels/cpu/trace_kernel.cc；GPU kernel 实现位于 Paddle/paddle/phi/kernels/gpu/trace_kernel.cu。

TraceKernel 的 CPU 实现示例：

```cpp
template <typename T, typename Context>
void TraceKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 int offset,
                 int axis1,
                 int axis2,
                 DenseTensor* out) {
  auto* out_data = dev_ctx.template Alloc<T>(out);

  const DenseTensor diag =
      funcs::Diagonal<T, Context>(dev_ctx, &x, offset, axis1, axis2);
  if (diag.numel() > 0) {
    auto x = phi::EigenMatrix<T>::Reshape(diag, diag.dims().size() - 1);
    auto output = phi::EigenVector<T>::Flatten(*out);
    auto reduce_dim = Eigen::array<int, 1>({1});
    output.device(*dev_ctx.eigen_device()) = x.sum(reduce_dim);
    out->Resize(out->dims());
  } else {
    std::fill(out_data, out_data + out->numel(), static_cast<T>(0));
  }
}
```

说明：对于 kernel 内部临时使用的 DenseTensor 目前推荐使用 Empty、EmptyLike、Full 和 FullLike 接口进行创建。

**实现反向 Kernel 函数**

反向 kernel 的实现与前向是类似的。相关文件：paddle/phi/kernels/trace_grad_kernel.h、paddle/phi/kernels/cpu/trace_grad_kernel.cc、paddle/phi/kernels/gpu/trace_grad_kernel.cu。

**公共函数管理**

如果有一些函数会被多个 kernel 调用，可以创建非 kernel 的文件管理代码：
- 仅有当前 kernel 使用的辅助函数，和 kernel 实现放到同一个设备文件夹中
- 有同设备多个 kernel 使用的辅助函数，在 kernel 所在的设备目录创建 .h 放置代码
- 有跨设备多个 kernel 使用的辅助函数，在 kernels/funcs 目录下创建 .h/cc/cu 管理代码

#### 4.2.3 注册 Kernel 函数

在对应的 kernel 实现代码中添加注册 kernel 函数：

```cpp
PD_REGISTER_KERNEL(trace,
                   CPU,
                   ALL_LAYOUT,
                   phi::TraceKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
```

字段说明：
- trace：kernel 名称，和算子的名称一致
- CPU：backend 名称，一般主要就是 CPU 和 GPU
- ALL_LAYOUT：kernel 支持的 Tensor 布局，一般为 ALL_LAYOUT
- phi::TraceKernel：kernel 的函数名称，记得带上 namespace phi
- 剩余的均为 kernel 支持的数据类型

注意：
- 如果忘记添加注册相关的头文件，会给出一个 error: expected constructor, destructor, or type conversion before '(' token 的错误
- phi 下的注册宏后边是带函数体 { }，不是直接加分号
- 注册 kernel 的宏声明需要在 global namespace

## 五、封装 Python API

飞桨框架会对新增的算子 kernel 自动绑定 Python，开发者需要在 Python 端定义相应的 API。

详见：新增 Python 层 API（references/new_python_api.md）

## 六、添加单元测试

单测文件存放路径和命名方式：在 test/legacy_test/ 目录下，把对 Python API 的单元测试和 C++ 算子的单元测试写在同一个文件中，以 test_xxx_op.py 的形式命名（假设算子名为 xxx）。

### 6.1 C++ 算子单元测试

算子单元测试继承自 test/legacy_test/op_test.py 中的 OpTest 类。测试要点：
1. 在 setUp 函数定义输入、输出、属性参数，并生成随机输入数据
2. 在 Python 脚本中实现与前向算子相同的计算逻辑，与算子输出对比
3. 反向计算已自动集成进测试框架

注意：单测中的测试用例需要尽可能地覆盖 kernel 中的所有分支。

```python
class TestTraceOp(OpTest):
    def setUp(self):
        self.op_type = "trace"
        self.python_api = paddle.trace
        self.init_dtype()
        self.case = np.random.randn(20, 6).astype(self.dtype)
        self.inputs = {'Input': self.case}
        self.attrs = {'offset': 0, 'axis1': 0, 'axis2': 1}
        self.outputs = {'Out': np.trace(self.case)}

    def init_dtype(self):
        self.dtype = np.float64

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad(self):
        self.check_grad(['Input'], 'Out', check_pir=True)
```

关键点：
- self.op_type：算子名称，与 YAML 配置中的算子名一致
- self.python_api：对应的 Python API 函数
- self.inputs：输入数据，字典格式，key 与 YAML 配置中的输入名对应
- self.attrs：属性参数
- self.outputs：期望输出，用于与算子实际输出对比
- check_pir=True：开启 PIR 模式单测

### 6.2 Python API 单元测试

详见：新增 Python 层 API - 添加单元测试（references/new_python_api.md#五添加单元测试）

### 6.3 编译并运行单测（每次修改均需执行）

单测编写完成后，按以下命令验证执行：

```bash
cd ${ROOT_DIR}/Paddle/build
cmake .. && make -j$(nproc) > compile.log 2>&1
python test_xxx_op.py
```

根据报错信息修改代码，确保所有测试用例通过。每次修改后均需要重新执行本步骤。

**编译注意事项**：
- 无需重装，直接生效（勿执行 setup/install 等安装操作）
- 勿删除 build 目录（否则增量编译失效，编译时间极长）

## 七、新增 API 英文文档

详见：新增 Python 层 API - 新增 API 英文文档（references/new_python_api.md#六新增-api-英文文档）

## 八、开发算子注意事项

### 8.1 报错检查

实现算子时检查数据的合法性需要使用 PADDLE_ENFORCE 以及 PADDLE_ENFORCE_EQ 等宏定义：

```
PADDLE_ENFORCE(表达式, 错误提示信息)
PADDLE_ENFORCE_EQ(比较对象 A, 比较对象 B, 错误提示信息)
```

如果表达式为真，或者比较对象 A=B，则检查通过，否则会终止程序运行。

总体原则：任何使用了 PADDLE_ENFORCE 与 PADDLE_ENFORCE_XX 检查的地方，必须有详略得当的备注解释，错误提示信息不能为空。

报错提示信息书写建议：
1. 哪里错了？为什么错了？例如：ValueError: Mismatched label shape
2. 期望的输入是什么样的？实际的输入是怎样的？例如：Expected labels dimension=1. Received 4.
3. 能否给出修改意见？

更详细的报错检查规范请参考：《Paddle 报错信息文案书写规范》（../style_guide_and_references/error_message_writing_specification_cn.md）

### 8.2 算子兼容性问题

对算子的修改需要考虑兼容性问题，要保证算子修改之后，之前的模型都能够正常加载及运行。兼容性要求如下：
- 算子当前的所有输入输出参数不能被修改或删除
- 可以新增参数，但新增的 Tensor 类型变量需要设置为 optional
- 新增的非 Tensor 变量需要设置默认值

### 8.3 显存优化

**为可原位计算的算子注册 inplace**

有些算子的计算逻辑中，输出可以复用输入的显存空间。对于这类算子，可以注册 inplace，从而让框架在运行时自动地进行显存优化。

注册方式为在算子的 YAML 配置中添加 inplace 配置项，格式如：(x -> out)。示例：

```yaml
- op : reshape
  args : (Tensor x, IntArray shape)
  output : Tensor(out)
  ...
  inplace : (x -> out)
```

**减少反向算子中的无关变量**

通常反向算子会依赖于前向算子的某些输入、输出 Tensor。若反向算子只需要使用前向算子中输入和输出变量的 Shape 和 LoD 信息，但不依赖于变量中 Tensor 的内存 Buffer 数据，则可以通过 no_need_buffer 对该变量进行配置。示例：

```yaml
- backward_op : trace_grad
  forward : trace (Tensor x, int offset, int axis1, int axis2) -> Tensor(out)
  args : (Tensor x, Tensor out_grad, int offset, int axis1, int axis2)
  output : Tensor(x_grad)
  ...
  no_need_buffer : x
```

### 8.4 性能优化

- 第三方库选择：优先使用 cudnn、mkldnn、mklml、eigen 等高性能库，但需做 benchmark 验证
- CUDA Kernel 优化：减少 CUDA Kernel 调用次数，将多个小 Kernel 合并；减少 CPU 与 GPU 之间的拷贝和同步操作
- 更多优化方法参考：算子性能优化方法介绍（../op_optimization/op_optimization_method_introduction_cn.html）

### 8.5 稀疏梯度参数更新方法

目前稀疏梯度在做更新的时候会先对梯度做 merge，即对相同参数的梯度做累加，然后做参数以及附加参数（如 velocity）的更新。

### 8.6 混合设备调用

由于 GPU 是异步执行的，当 CPU 调用返回之后，GPU 端可能还没有真正的执行，所以如果在算子中创建了 GPU 运行时需要用到的临时变量，当 GPU 开始运行的时候，该临时变量可能在 CPU 端已经被释放，这样可能会导致 GPU 计算出错。

关于 GPU 中的一些同步和异步操作：Kernel launches、Memory copies within a single device's memory、Memory copies from host to device of a memory block of 64 KB or less、Memory copies performed by functions that are suffixed with Async、Memory set function calls 都是异步的。

关于 cudaMemCpy 和 cudaMemCpyAsync 注意事项：
- 如果数据传输是从 GPU 端到非页锁定的 CPU 端，数据传输将是同步，即使调用的是异步拷贝操作
- 如果数据传输是从 CPU 端到 CPU 端，数据传输将是同步的，即使调用的是异步拷贝操作

### 8.7 算子数值稳定性问题

有些算子存在数值稳定性问题，主要原因是在多次运行时，对浮点型数据施加操作的顺序可能不同。GPU 是通过多线程并行计算的方式来加速计算的，所以很容易出现对浮点数施加操作的顺序不固定现象。

目前发现 cudnn 中的卷积操作、cudnn 中的 MaxPooling、CUDA 中 CudaAtomicXX、ParallelExecutor 的 Reduce 模式下参数梯度的聚合等操作运行结果是非确定的。

Paddle 中添加了一些 FLAGS，比如使用 FLAGS_cudnn_deterministic 来强制 cudnn 使用确定性算法、FLAGS_cpu_deterministic 强制 CPU 端的计算使用确定性方法。

### 8.8 算子的数学公式

如果算子有数学公式，一定要在代码中将数学公式写明，并在 Python API 的 Doc 中显示，因为用户在对比不同框架的计算结果时可能需要了解 Paddle 对算子是怎么实现的。

### 8.9 LoD 在算子内部的传导规范

根据算子是否依赖 LoD，分为两类：
- LoD-transparent：计算不依赖 LoD，如 conv2d_op、batch_norm_op
- LoD-Based：计算依赖 LoD，如 lstm_op、gru_op、sequence_ops

前向传导：对于"不变"和"改变"两种情况，需在 InferMeta 中调用 ShareLoD() 进行传导。

反向传导：输入 Var 对应的梯度 GradVar 的 LoD 应与 Var 自身相同，直接共享即可。

### 8.10 PyTorch 对齐关键配置文件

新增 OP 时，除标准流程外，PyTorch 对齐还需特别关注：
- op_compat.yaml：C++ 层参数名兼容性配置，支持 dim↔axis、input↔x 等参数名互换
- python_api_info.yaml：Python API 名称注册，注册 paddle.xxx 和 paddle.Tensor.xxx 两种访问路径

### 8.11 多输出 OP 注意事项

YAML 配置、InferMeta 和 Kernel 函数签名中，多输出 OP 需为每个输出分别命名并传递指针。

## 九、更多信息

### 9.1 Paddle 基于 Yaml 配置自动生成算子代码的逻辑解读

Paddle 支持动态图和静态图两种模式，在 YAML 配置文件中完成算子基本属性的定义后，需要进行解析并分别生成动态图和静态图所对应的算子代码逻辑。

架构说明：当开发者添加一个新的 C++ 算子时，只需要完成 Kernel、算子定义 Yaml 配置文件和 Python API 三个部分的代码开发，其余部分都会通过自动代码生成来完成。

开发者需要开发的部分：
- Python API（python 层）
- YAML 配置文件（ops.yaml）
- Kernel 实现（CPU/GPU）

自动代码生成部分：
- 动态图：Python-C 接口 -> Autograd API -> C++ API
- 静态图：OpMaker 注册 -> REGISTER_OPERATOR 等注册组件

动态图中自动生成的代码包括从 Python API 到计算 Kernel 间的各层调用接口实现：
- C++ API：一套与 Python API 参数对齐的 C++ 接口。前向算子生成的 C++ API 代码位于 `paddle/phi/api/include/` 目录下的 `api.h` 和 `api.cc`；反向算子生成的 C++ API 代码位于 `paddle/phi/api/backward/` 目录下的 `backward_api.h` 和 `backward_api.cc`
- 动态图前向函数与反向节点（Autograd API）：在 C++ API 的基础上进行了封装，生成的相关代码位于 `paddle/fluid/eager/api/generated/eager_generated/forwards/` 目录下的 `dygraph_functions.h` 和 `dygraph_functions.cc`
- Python-C 函数：将支持自动微分功能的 C++ 的函数接口暴露到 Python 层，生成的 Python-C 接口代码位于 `paddle/fluid/pybind/` 目录下的 `eager_op_function.h` 和 `eager_op_function.cc`

静态图的执行流程与动态图不同，Python API 主要负责组网，算子的调度和 kernel 计算由静态图执行器来完成，自动生成的代码是将配置文件中的算子信息注册到框架内供执行器调度，主要包括 Paddle/paddle/fluid/framework/op_proto_maker.h 中的 OpMaker 和 REGISTER_OPERATOR 等静态图算子注册组件。
