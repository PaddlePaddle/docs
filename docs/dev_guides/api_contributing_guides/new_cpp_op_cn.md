# C++ OP 开发

> 注：飞桨原生算子的开发范式正处在重构升级后的上线初期，如果在开发过程中遇到问题欢迎通过[issue](https://github.com/PaddlePaddle/Paddle/issues)向我们反馈。

## 1. 概念简介

本教程对新增原生算子的方法进行介绍，首先新增一个算子大概需要以下几个步骤：

1. 新增算子描述及定义：描述前反向算子的输入、输出、属性，实现InferMeta函数
2. 新增算子Kernel：实现算子在各种设备上的计算逻辑
3. 封装Python API：封装Python端调用算子的接口
4. 添加单元测试：验证新增算子的正确性

以上4个步骤添加的文件，在Paddle中的位置如下（假设算子名为`xxx`）：

<table>
<thead>
<tr>
<th>内容</th>
<th>新增文件位置</th>
</tr>
</thead>
<tbody>
<tr>
<td>算子描述及定义</td>
<td>python/paddle/utils/code_gen/api.yaml & python/paddle/utils/code_gen/backward.yaml</td>
</tr>
<tr>
<td>算子InferMeta</td>
<td>paddle/phi/infermeta目录下的相应文件中</td>
</tr>
<tr>
<td>算子kernel</td>
<td>paddle/phi/kernels/xxx_kernel.h & xxx_kernel.cc & xxx_grad_kernel.h & xxx_grad_kernel.cc（一般情况）</td>
</tr>
<tr>
<td>Python API</td>
<td>python/paddle目录下的相应子目录中</td>
</tr>
<tr>
<td>单元测试</td>
<td>python/paddle/fluid/tests/unittests/test_xxx_op.py</td>
</tr>
</tbody>
</table>

关于Python API所处位置，可以参考 [飞桨官方 API 文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/index_cn.html) ，了解各个目录存放API的性质，从而决定具体的放置目录。

接下来，我们以Trace操作，计算输入 Tensor 在指定平面上的对角线元素之和，并输出相应的计算结果，即 [trace](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/trace_cn.html#trace) 为例来介绍如何新增算子。

## 2. 新增算子描述及定义

算子描述及定义是定义运算的基本属性，主要包括算子的输入、输出以及各项非计算逻辑的配置，这些都是设备无关的。

### 2.1 算子Yaml文件配置
我们在`python/paddle/utils/code_gen/api.yaml`和`python/paddle/utils/code_gen/backward.yaml`文件中对算子进行描述及定义，在框架编译时会根据Yaml文件中的配置自动生成C++端的相关代码接口以及内部实现（可参考[Paddle基于Yaml配置的算子代码自动生成](new_cpp_op_notes_cn.md)），下面主要以Trace为例介绍算子的Yaml配置规则：

python/paddle/utils/code_gen/api.yaml：
```
- api : trace
  args : (Tensor x, int offset = 0, int axis1 = 0, int axis2 = 1)
  output : Tensor(out)
  infer_meta :
    func : TraceInferMeta
  kernel :
    func : trace
  backward : trace_grad
```
python/paddle/utils/code_gen/backward.yaml：
```
- backward_api : trace_grad
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

`api.yaml`和`backward.yaml`分别对算子的前向和反向进行配置，首先`api.yaml`中前向算子的配置规则如下：
<table>
<thead>
<tr>
<th>配置项</th>
<th>配置内容及规则</th>
</tr>
</thead>
<tbody>
<tr>
<td>api</td>
<td>算子名称，与该算子Python API函数名相同（命名方式为：全小写+下划线），示例中为trace</td>
</tr>
<tr>
<td>args</td>
<td>算子输入参数，与该算子Python API函数的输入参数对应（当前支持的输入数据类型包括：Tensor, Tensor[]/*Tensor数组*/, float, double, bool, int, int64_t, int[], int64_t[], str, Place, DataType, DataLayout, IntArray/*主要用于表示shape,index和axes等类型数据，可以直接使用Tensor或者普通整型数组构造，目前仍在测试阶段，如非必要暂不建议使用*/, Scalar/*标量，支持不同的普通数据类型*/）</td>
</tr>
<tr>
<td>output</td>
<td>算子输出类型，目前支持Tensor和Tensor[], 多个输出间用逗号“,”分隔开。可以使用”()”选择性标记输入的名字, 如未标记默认为'out'</td>
</tr>
<tr>
<td>infer_meta</td>
<td>InferMeta函数负责根据输入推断返回Tensor的维度与类型，这里是对算子使用的InferMeta函数进行配置</td>
</tr>
<tr>
<td>infer_meta:func</td>
<td>调用的InferMeta函数, 这里trace调用的是TraceInferMeta函数</td>
</tr>
<tr>
<td>infer_meta:param</td>
<td>InferMeta函数的输入参数，可以对args中的参数进行选择传入，未配置则默认传入args中的所有参数，示例中未配置本项，所以传入的参数为[x, offset, axis1, axis2]。output项中的参数作为输出无需配置会自动传入InferMeta函数中</td>
</tr>
<tr>
<td>kernel</td>
<td>算子的计算Kernel配置</td>
</tr>
<tr>
<td>kernel:func</td>
<td>算子对应kernel函数的注册名</td>
</tr>
<tr>
<td>kernel:param</td>
<td>kernel函数的输入参数，配置规则与InferMeta函数的param配置相同</td>
</tr>
<tr>
<td>kernel:data_type</td>
<td>根据指定参数推导调用kernel的data_type类型（对应kernel函数的模板参数'T'），默认不进行配置，会根据输入Tensor自动进行推导。如果kernel的data_type类型由某个输入的Tensor决定，需要将该Tensor参数的变量名填入该项。示例中未配置则kernel的data_type由输入变量'x'决定</td>
</tr>
<td>kernel:backend</td>
<td>根据指定参数来选择调用kernel的Backend（Kernel执行的具体设备，如CPU、GPU等），默认不进行配置，会根据输入Tensor自动进行推导。如果kernel执行的backend类型由某个输入的Tensor决定，需要将该Tensor参数的变量名填入该项。示例中未配置则kernel执行的Backend与输入变量'x'的Backend相同</td>
</tr>
<tr>
<td>backward</td>
<td>算子对应的反向算子名称，如果没有反向则不需要配置，示例中trace算子的反向为trace_grad</td>
</tr>
<tr>
<td colspan="2">特殊配置项（目前特殊配置项还处于不稳定阶段，后续可能会有调整更新）</td>
</tr>
<tr>
<td>optional</td>
<td>指定输入Tensor为可选输入，用法可参考dropout中seed_tensor(python/paddle/utils/code_gen/legacy_api.yaml中)</td>
</tr>
<tr>
<td>inplace</td>
<td>算子对指定的输入做原位处理并作为输出结果返回，使用格式：(x -> out)，具体用法可参考relu算子<br>
特殊规则：如果api中算子名称有'_'后缀则只生成支持inplace功能的接口，如果算子名称没有'_'后缀，则会同时生成支持inplace操作的接口(自动添加'_'后缀)和不支持inplace的普通接口共两套接口
</td>
</tr>
<tr>
<td>view</td>
<td>与inplace机制类似，区别在于view模式返回的结果只是与输入共享内存，并不是输入Tensor变量本身，使用格式：(x -> out)，具体用法可参考reshape算子</td>
</tr>
<tr>
<td>intermediate</td>
<td>标记前向计算中输出的用于反向计算的中间变量，不会出现在Python API的返回结果中，相关设计正在完善中，新增算子时不建议使用</td>
</tr>
</tbody>
</table>


`backward.yaml`中反向算子的配置规则如下：
<table>
<thead>
<tr>
<th>配置项</th>
<th>配置内容及规则</th>
</tr>
</thead>
<tbody>
<tr>
<td>backward_api</td>
<td>反向算子名称，一般命名方式为：前向算子名称+'_grad'，二阶算子则为前向算子名称+'_double_grad'</td>
</tr>
<tr>
<td>forward</td>
<td>对应前向算子的名称、参数、返回值，需要与api.yaml中前向算子配置一致</td>
</tr>
<tr>
<td>args</td>
<td>反向算子输入参数, 示例中'x'表示将前向的'x'变量输入到反向，'out_grad'表示前向输出'out'对应的反向梯度<br>
约束条件1：所有参数需要在forward配置项的参数中（输入、输出以及输出对应的反向梯度）找到对应（根据变量名匹配）<br>
约束条件2：反向输入参数需要以：a.前向输入Tensor b.前向输出Tensor c.前向输出Tensor的反向梯度 d.前向非Tensor类型属性变量 的顺序排列，只需添加反向计算需要用到的前向参数<br>
</td>
</tr>
<tr>
<td>output</td>
<td>反向算子输出，顺序需要与前向输入Tensor一致，比如前向输入(Tensor x, Tensor y)，则反向输出必须为Tensor(x_grad), Tensor(y_grad)</td>
</tr>
<tr>
<td>infer_meta</td>
<td>与前向配置规则相同</td>
</tr>
<tr>
<td>kernel</td>
<td>与前向配置规则相同</td>
</tr>
<tr>
<td>backward</td>
<td>反向算子对应的更高阶反向算子名称，如一阶反向算子的反向为二阶反向算子</td>
</tr>
<tr>
<td colspan="2">特殊配置项（目前特殊配置项还处于不稳定阶段，后续可能会有调整更新）</td>
</tr>
<tr>
<td>no_need_buffer</td>
<td>可选配置，标记的Tensor变量在前向运行完成后，持有的内存或显存会被释放，以减少训练过程中的内存使用。trace_grad由于反向算子只需要前向变量'x'的维度信息，不需要内存数据，所以可以标记为no_need_buffer提前释放内存<br>
注意：由于Tensor内存被释放后会影响dtype接口的使用，所以需要在kernel的data_type配置项中指定其他的Tensor来推导kernel的data_type</td>
</tr>
<tr>
<td>optional</td>
<td>与前向配置规则相同</td>
</tr>
<tr>
<td>inplace</td>
<td>与前向配置规则相同</td>
</tr>
</tbody>
</table>

### 2.2 实现InferMeta函数

`InferMeta`函数是根据输入参数，推断算子输出Tensor基本信息的函数，推断的信息包括输出Tensor的`shape`、`data type`及`data layout`，同时它也承担了检查输入数据维度、类型等是否合法的功能。

[TraceOp的InferMeta函数](https://github.com/PaddlePaddle/Paddle/blob/befa78ea3fa9d0dae096a7de91f626b0c31daee8/paddle/phi/infermeta/unary.cc#L721) 实现如下：

```cpp
void TraceInferMeta(
    const MetaTensor& x, int offset, int axis1, int axis2, MetaTensor* out) {
  int dim1 = axis1;
  int dim2 = axis2;

  auto x_dims = x.dims();

  int dim1_ = dim1 < 0 ? x_dims.size() + dim1 : dim1;
  int dim2_ = dim2 < 0 ? x_dims.size() + dim2 : dim2;

  PADDLE_ENFORCE_GE(
      x_dims.size(),
      2,
      phi::errors::OutOfRange(
          "Input's dim is out of range (expected at least 2, but got %ld).",
          x_dims.size()));
  PADDLE_ENFORCE_LT(
      dim1_,
      x_dims.size(),
      phi::errors::OutOfRange(
          "Attr(dim1) is out of range (expected to be in range of [%ld, "
          "%ld], but got %ld).",
          -(x_dims.size()),
          (x_dims.size() - 1),
          dim1));
  PADDLE_ENFORCE_LT(
      dim2_,
      x_dims.size(),
      phi::errors::OutOfRange(
          "Attr(dim2) is out of range (expected to be in range of [%ld, "
          "%ld], but got %ld).",
          -(x_dims.size()),
          (x_dims.size() - 1),
          dim2));
  PADDLE_ENFORCE_NE(
      dim1_,
      dim2_,
      phi::errors::InvalidArgument("The dimensions should not be identical "
                                   "%ld vs %ld.",
                                   dim1,
                                   dim2));

  auto sizes = vectorize(x_dims);
  if (x_dims.size() == 2) {
    sizes.clear();
    sizes.push_back(1);
  } else {
    sizes.erase(sizes.begin() + std::max(dim1_, dim2_));
    sizes.erase(sizes.begin() + std::min(dim1_, dim2_));
  }
  out->set_dims(phi::make_ddim(sizes));
  out->set_dtype(x.dtype());
}
```

其中，`MetaTensor`是对底层异构Tensor的抽象封装，仅支持对底层Tensor的维度、数据类型、布局等属性进行读取和设置，具体方法请参考 [meta_tensor.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/core/meta_tensor.h)。

**InferMeta的实现位置**

InferMeta的文件放置规则（以Tensor输入个数为判定标准）：

- `nullary.h`：没有输入Tensor参数的函数
- `unary.h`：仅有一个输入Tensor参数的函数
- `binary.h`：有两个输入Tensor参数的函数
- `ternary.h`：有三个输入Tensor参数的函数
- `multiary.h`：有三个以上输入Tensor或者输入为`vector<Tensor>`的函数
- `backward.h`：反向op的InferMeta函数一律在此文件中，不受前序规则限制

**InferMeta的编译时与运行时**

在我们的静态图网络中，`InferMeta`操作在[编译时(compile time)和运行时(run time)](https://github.com/PaddlePaddle/FluidDoc/blob/release/1.2/doc/fluid/getstarted/Developer's_Guide_to_Paddle_Fluid.md#%E8%AE%A9%E6%88%91%E4%BB%AC%E5%9C%A8fluid%E7%A8%8B%E5%BA%8F%E5%AE%9E%E4%BE%8B%E4%B8%AD%E5%8C%BA%E5%88%86%E7%BC%96%E8%AF%91%E6%97%B6%E5%92%8C%E8%BF%90%E8%A1%8C%E6%97%B6)都会被调用，在compile time时，由于真实的维度未知，框架内部用-1来表示，在run time时，用实际的维度表示，因此维度的值在compile time和 run time时可能不一致，如果存在维度的判断和运算操作，InferMeta就需要区分compile time 和 run time。

对于此类InferMeta函数，需要在函数声明的参数列表末尾增加 `MetaConfig` 参数，例如：

```
void ConcatInferMeta(const std::vector<MetaTensor*>& x,
                     const Scalar& axis_scalar,
                     MetaTensor* out,
                     MetaConfig config = MetaConfig());
```

然后在函数体中，使用 `config.is_runtime` 判断出于编译时还是运行时。

具体地，以下两种情况需要区分compile time和 run time。

1. 检查

    如以下代码：

    ```cpp
    int i = xxx;
    PADDLE_ENFORCE_GT(x.dims()[i] , 10)
    ```

    在compile time的时候，x.dims()[i]可能等于-1，导致这个PADDLE_ENFORCE_GT报错退出。

    如果用了以下paddle中定义的宏进行判断：

    ```cpp
    PADDLE_ENFORCE_EQ (x.dims()[i] , 10)
    PADDLE_ENFORCE_NE (x.dims()[i] , 10)
    PADDLE_ENFORCE_GT (x.dims()[i] , 10)
    PADDLE_ENFORCE_GE (x.dims()[i] , 10)
    PADDLE_ENFORCE_LT (x.dims()[i] , 10)
    PADDLE_ENFORCE_LE (x.dims()[i] , 10)
    ```

    都需要注意区分compile time和run time

2. 运算

    如以下代码:
    ```cpp
    auto x_dim = x.dims();
    int i = xxx;
    y_dim[0] = x_dim[i] + 10
    ```

    在compile time的时候，x_dim[i]可能等于-1，得到的 y_dim[0] 等于 9，是不符合逻辑的

    如果用到了类似以下的运算操作

    ```cpp
    y_dim[i] = x_dim[i] + 10
    y_dim[i] = x_dim[i] - 10
    y_dim[i] = x_dim[i] * 10
    y_dim[i] = x_dim[i] / 10
    y_dim[i] = x_dim[i] + z_dim[i]
    ```

    都需要区分compile time和run time

3. 处理的标准

    - 检查： compile time的时候不判断维度等于-1的情况，但在runtime的时候检查
    - 运算： -1和其他数做任何运算都要等于-1

4. 参考代码

    （1） 判断的实现方法可以参考 [SigmoidCrossEntropyWithLogitsInferMeta](https://github.com/PaddlePaddle/Paddle/blob/cd28cddbfb5f5643947291e9a640ecd414dc8dae/paddle/phi/infermeta/binary.cc#L650)，SigmoidCrossEntropyWithLogits 要求X和labels的两个输入，除了最后一维以外，其他的维度完全一致

    ```cpp
      bool check = true;
      if ((!config.is_runtime) &&
          (phi::product(x_dims) <= 0 || phi::product(labels_dims) <= 0)) {
        check = false;
      }

      if (check) {
        PADDLE_ENFORCE_EQ(
            phi::slice_ddim(x_dims, 0, rank),
            phi::slice_ddim(labels_dims, 0, rank),
            phi::errors::InvalidArgument(
                "Input(X) and Input(Label) shall have the same shape "
                "except the last dimension. But received: the shape of "
                "Input(X) is [%s], the shape of Input(Label) is [%s].",
                x_dims,
                labels_dims));
      }
    ```

    （2） 运算的实现可以参考 [ConcatInferMeta](https://github.com/PaddlePaddle/Paddle/blob/0604df9e70dfe7be8a21df6a80d9fa6d4939bd9d/paddle/phi/infermeta/multiary.cc#L323)，concat在InferShape判断时，调用`ComputeAndCheckShape`，除了进行concat轴之外，其他的维度完全一致；在生成output的维度时，把concat轴的维度求和，其他的维度和输入保持一致。

    ```cpp
      const size_t n = inputs_dims.size();
      auto out_dims = inputs_dims[0];
      size_t in_zero_dims_size = out_dims.size();
      for (size_t i = 1; i < n; i++) {
        PADDLE_ENFORCE_EQ(
            inputs_dims[i].size(),
            out_dims.size(),
            phi::errors::InvalidArgument("The shape of input[0] and input[%d] "
                                        "is expected to be equal."
                                        "But received input[0]'s shape = "
                                        "[%s], input[%d]'s shape = [%s].",
                                        i,
                                        inputs_dims[0],
                                        i,
                                        inputs_dims[i]));
        for (size_t j = 0; j < in_zero_dims_size; j++) {
          if (j == axis) {
            if (is_runtime) {
              out_dims[axis] += inputs_dims[i][j];
            } else {
              if (inputs_dims[i][j] == -1 || out_dims[j] == -1) {
                out_dims[axis] = -1;
              } else {
                out_dims[axis] += inputs_dims[i][j];
              }
            }
          } else {
            bool check_shape =
                is_runtime || (inputs_dims[0][j] > 0 && inputs_dims[i][j] > 0);
            if (check_shape) {
              // check all shape in run time
              PADDLE_ENFORCE_EQ(inputs_dims[0][j],
                                inputs_dims[i][j],
                                phi::errors::InvalidArgument(
                                    "The %d-th dimension of input[0] and input[%d] "
                                    "is expected to be equal."
                                    "But received input[0]'s shape = "
                                    "[%s], input[%d]'s shape = [%s].",
                                    j,
                                    i,
                                    inputs_dims[0],
                                    i,
                                    inputs_dims[i]));
            }
            if (!is_runtime && out_dims[j] == -1 && inputs_dims[i][j] > 0) {
              out_dims[j] = inputs_dims[i][j];
            }
          }
        }
      }
    ```

## 3. 新增算子Kernel

新增算子Kernel在 `paddle/phi/kernels` 目录中完成

### 3.1 kernels目录结构

`paddle/phi/kernels` 基本目录结构如下

```
paddle/phi/kernels
./ (根目录放置设备无关的kernel声明和实现)
./cpu（仅放置cpu后端的kernel实现）
./gpu（仅放置gpu后端的kernel实现）
./xpu（仅放置百度kunlun后端的kernel实现）
./gpudnn
./funcs（放置一些支持多设备的、在多个kernel中使用的公共functor和functions）
...
```

一般情况下，新增算子仅需要关注kernels根目录及kernel所支持设备的子目录即可：

- kernels 根目录，放置设备无关的kernel.h和kernel.cc
  - 例如，一个kernel除了一些简单的设备无关的C++逻辑，关键计算逻辑均是复用已有的phi kernel函数实现的，那么这个kernel实现是天然能够适配所有设备及后端的，所以它的声明和实现均直接放置到kernels目录下即可
- kernels下一级子目录，原则上按照backend分类按需新建，放置特定后端的kernel实现代码

下面给出两种典型kernel新增时文件放置位置的说明：

1. 新增与设备无关的Kernel

    该类Kernel 实现与所有硬件设备无关，只需要一份代码实现，可参考reshape kernel。其新增文件及目录包括：

    - `paddle/phi/kernels/xxx_kernel.h`
    - `paddle/phi/kernels/xxx_kernel.cc`

    如果是反向kernel，则使用 `grad_kernel` 后缀即可：

    - `paddle/phi/kernels/xxx_grad_kernel.h`
    - `paddle/phi/kernels/xxx_grad_kernel.cc`

2. 新增与设备相关、且CPU&GPU分别实现的Kernel

    还有部分Kernel的实现，CPU 和GPU 上逻辑不同，此时没有共同实现的代码，需要区分CPU和GPU 硬件。
    CPU 的实现位于`paddle/phi/kernels/cpu` 目录下； GPU的实现位于`paddle/phi/kernels/gpu` 下，可参考dot kernel，cast kernel等。其新增文件及目录包括：

    - `paddle/phi/kernels/xxx_kernel.h`
    - `paddle/phi/kernels/cpu/xxx_kernel.cc`
    - `paddle/phi/kernels/gpu/xxx_kernel.cu`

    相应地，反向kernel新增文件为：

    - `paddle/phi/kernels/xxx_grad_kernel.h`
    - `paddle/phi/kernels/cpu/xxx_grad_kernel.cc`
    - `paddle/phi/kernels/gpu/xxx_grad_kernel.cu`

### 3.2 Kernel 写法

#### 3.2.1 声明 Kernel 函数

- 以trace op为例，首先在`paddle/phi/kernels`目录下新建 [`trace_kernel.h`](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/trace_kernel.h) 文件，用于放置前向Kernel函数声明。

> 注：Kernel函数声明的参数列表原则上与Python API参数列表一致

```
template <typename T, typename Context>
void TraceKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 int offset,
                 int axis1,
                 int axis2,
                 DenseTensor* out);
```

> 注：所有的kernel声明，统一放在namespace phi中，缩短函数的调用前缀使调用写法更加简洁

说明如下：

1. 模板为固定写法，第一个模板参数为数据类型`T`，第二个模板参数为设备上下文`Context`，`template <typename T, typename Context>`
2. 函数命名：Kernel 的命名统一加Kernel 后缀。即：Kernel名称+Kernel 后缀，驼峰式命名，例如：AddKernel
3. 参数顺序：Context， InputTensor …, Attribute …, OutTensor* 。即：第一位参数为Context， 后边为输入的Tensor， 接着是输入的属性参数， 最后是输出的Tensor的指针参数。如果Kernel没有输入Tensor 或者没有属性参数，略过即可
2. 第1个函数参数，类型为 `const Context&` 的dev_ctx
3. 第2个函数参数，输入Tensor，类型一般为 `const DenseTensor&`
4. 第3-5个函数参数，均为attribute（根据具体的含义，选择特定的int，float，vector<int>等类型），多个attribute 可以参考python端API定义的顺序，变量命名对齐python api
5. 第6个函数参数，输出Tensor，类型一般为`DenseTensor*`，多个output 可以参考python端API定义的顺序， 变量命名对齐python api

> **特殊情况说明：**
> 1. **特殊模板参数**：对于某些Kernel （如reshape ，copy），这些kernel不关注数据类型T， 可以省去第一个模板参数，即为：`template <typename Context>`
> 2. **特殊输入类型**：对于某些特殊Kernel （如concat 和split kernel）的部分输入或输出是数组类型的DenseTensor（OpMaker中有`AsDuplicable`标记）, 此时输入类型为：`const std::vector<const DenseTensor*>&`; 输出类型为：`std::vector<DenseTensor*>`

#### 3.2.2 实现 Kernel 函数

此处trace op的kernel属于前述第2中情况，即CPU与GPU Kernel需要分别实现。

- cpu kernel实现位于：[paddle/phi/kernels/cpu/trace_kernel.cc](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/cpu/trace_kernel.cc)
- gpu kernel实现位于：[paddle/phi/kernels/gpu/trace_kernel.cu](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/gpu/trace_kernel.cu)

下面为 `TraceKernel` 的cpu实现：

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

**Kernel复用：**

此处TraceKernel的实现并未复用其他Kernel，但如果有需要也是可以复用的，Kernel复用时，直接 include 相应Kernel头文件，在函数中调用即可，例如 [triangular_solve_kernel](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/cpu/triangular_solve_kernel.cc) 复用 empty和expand kernel。

首先在triangular_solve_kernel.cc头部include相应头文件：

```cpp
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/expand_kernel.h"
```

然后在Kernel实现中即可直接调用以上两个头文件中的Kernel，代码片段如下：

```cpp
  // Tensor broadcast to 'out' and temp 'x_bst'
  ScalarArray x_bst_dims(x_bst_dims_vec);
  DenseTensor x_bst = phi::Empty<T, Context>(dev_ctx, x_bst_dims);
  const T* x_bst_data = x_bst.data<T>();
  ExpandKernel<T, Context>(dev_ctx, x, x_bst_dims, &x_bst);
```

反向Kernel的实现与前向是类似的，此处不再赘述，可以直接参考前述对应链接中的代码实现。

**公共函数管理：**

如果有一些函数会被多个Kernel调用，可以创建非 kernel 的文件管理代码，规则如下：

1. 仅有当前kernel使用的辅助函数（具体到设备，比如trace的cpu kernel），一律和kernel实现放到同一个设备文件夹中
    - 如果辅助函数相关代码较少，就直接和kernel实现放到同一个`.cc/cu`中
    - 如果辅助函数相关代码较多，就在kernel所在的设备目录创建`.h`管理代码
2. 有同设备多个kernel使用的辅助函数，在kernel所在的设备目录创建`.h`放置代码
3. 有跨设备多个kernel使用的辅助函数，在`kernels/funcs`目录下创建`.h/cc/cu`管理代码
4. 如果当前依赖的辅助函数可以直接归类到`kernels/funcs`目录下已有的文件中，则直接放过去，不用创建新的文件

**反向Kernel参数映射函数添加**

现阶段，反向Kernel除了实现外，还需要添加一个参数映射函数。

仍然以trace op为例，首先在`paddle/phi/ops/compat`目录下新建`trace_sig.cc`文件，用于放置这里的映射函数。

- 由于函数式kernel的一个最重要的特别就是参数顺序和类型（顺序和类型是关键，变量名称不影响），我们需要定义一个函数来做一个从OpMaker中如何获取信息，并且按照顺序传递给新的kernel函数； 这个模块就是OpArgumentMapping， trace反向op的OpArgumentMapping定义如下， KernelSignature共包含4个内容
    1. kernel名称，这个是我们给kernel注册的时候的名称
    2. input list： 这个要和OpMaker（或者GradOpMaker）中定义的Key要完全一致
    3. attribute list： 这个要和OpMaker（或者GradOpMaker）中定义的Key要完全一致
    4. output list： 这个要和OpMaker（或者GradOpMaker）中定义的Key要完全一致


    ```cpp
    #include "paddle/phi/core/compat/op_utils.h"

    namespace phi {

    KernelSignature TraceGradOpArgumentMapping(const ArgumentMappingContext& ctx) {
      return KernelSignature("trace_grad",
                             {GradVarName("Out"), "Input"},
                             {"offset", "axis1", "axis2"},
                             {GradVarName("Input")});
    }

    }  // namespace phi

    PD_REGISTER_ARG_MAPPING_FN(trace_grad, phi::TraceGradOpArgumentMapping);
    ```

>注：没有input list或attribute list的，相应花括号内留空，不能省略花括号

#### 3.2.3 注册 Kernel 函数

注册kernel的方式比较简单，直接使用注册宏注册即可，示例如下：

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
1. `trace`: kernel名称，和Op的名称一致
2. `CPU`: backend名称， 一般主要就是CPU和GPU
3. `ALL_LAYOUT`: kernel支持的Tensor布局，一般为ALL_LAYOUT，及支持所有布局类型
4. `phi::TraceKernel`: kernel的函数名称，记得带上namespace phi
5. 剩余的均为Kernel支持的数据类型

> 注意：
> 1. 如果忘记添加注册相关的头文件，会曝出一个xx的错误，如果遇到，请检查include的头文件
> 2. phi下的注册宏后边是带函数体{ }，不是直接加分号，此处与旧的注册宏有小区别
> 3. 注册kernel的宏声明需要在global namespace

### 3.3 编译测试

实现完Op和Kernel之后，建议先编译测试一下，编译成功之后，再继续后面的步骤。

详细的编译环境准备和执行流程可参考[从源码编译](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/compile/fromsource.html)，下面简单介绍几个主要步骤。
在 `Paddle` 代码目录下创建并切换到build目录：

```
mkdir build && cd build
```

执行`cmake`命令，具体选项可参考[从源码编译](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/compile/fromsource.html)中的介绍，下面的命令为编译Python3.7，GPU版本，带测试，Release版本的Paddle。

```
cmake .. -DPY_VERSION=3.7 -DWITH_GPU=ON -DWITH_TESTING=ON -DCMAKE_BUILD_TYPE=Release
```

在`build`目录下，运行下面命令可以进行编译整个paddle：

```
make -j$(nproc)
```

**注意：**
新增op后请重新执行`cmake`命令，然后再执行`make`命令编译paddle。

## 4. 封装Python API

系统会对新增的Op即Kernel自动绑定Python，并链接到生成的lib库中，然后在Python端定义相应的API，在API内调用新增算子，并添加相应的中英文文档描述即可。

[`paddle.trace`](https://github.com/PaddlePaddle/Paddle/blob/bd4dc3be34584f9b273ecec07297fb05e1cf4c52/python/paddle/tensor/math.py#L2277) 的Python API实现位于 `python/paddle/tensor/math.py` 中，具体实现如下：

```python
def trace(x, offset=0, axis1=0, axis2=1, name=None):
    """
    **trace**

    This OP computes the sum along diagonals of the input tensor x.

    If ``x`` is 2D, returns the sum of diagonal.

    If ``x`` has larger dimensions, then returns an tensor of diagonals sum, diagonals be taken from
    the 2D planes specified by axis1 and axis2. By default, the 2D planes formed by the first and second axes
    of the input tensor x.

    The argument ``offset`` determines where diagonals are taken from input tensor x:

    - If offset = 0, it is the main diagonal.
    - If offset > 0, it is above the main diagonal.
    - If offset < 0, it is below the main diagonal.
    - Note that if offset is out of input's shape indicated by axis1 and axis2, 0 will be returned.

    Args:
        x(Tensor): The input tensor x. Must be at least 2-dimensional. The input data type should be float32, float64, int32, int64.
        offset(int, optional): Which diagonals in input tensor x will be taken. Default: 0 (main diagonals).
        axis1(int, optional): The first axis with respect to take diagonal. Default: 0.
        axis2(int, optional): The second axis with respect to take diagonal. Default: 1.
        name (str, optional): Normally there is no need for user to set this property. For more information, please refer to :ref:`api_guide_Name`. Default: None.

    Returns:
        Tensor: the output data type is the same as input data type.

    Examples:
        .. code-block:: python

            import paddle

            case1 = paddle.randn([2, 3])
            case2 = paddle.randn([3, 10, 10])
            case3 = paddle.randn([3, 10, 5, 10])
            data1 = paddle.trace(case1) # data1.shape = [1]
            data2 = paddle.trace(case2, offset=1, axis1=1, axis2=2) # data2.shape = [3]
            data3 = paddle.trace(case3, offset=-3, axis1=1, axis2=-1) # data2.shape = [3, 5]
    """
    def __check_input(input, offset, dim1, dim2):
        check_dtype(x.dtype, 'Input',
                    ['int32', 'int64', 'float16', 'float32', 'float64'],
                    'trace')

        input_shape = list(x.shape)
        assert len(input_shape) >= 2,                     \
                "The x must be at least 2-dimensional, "   \
                "But received Input x's dimensional: %s.\n" %  \
                len(input_shape)

        axis1_ = axis1 if axis1 >= 0 else len(input_shape) + axis1
        axis2_ = axis2 if axis2 >= 0 else len(input_shape) + axis2

        assert ((0 <= axis1_) and (axis1_ < len(input_shape))),     \
            "The argument axis1 is out of range (expected to be in range of [%d, %d], but got %d).\n"  \
            % (-(len(input_shape)), len(input_shape) - 1, axis1)

        assert ((0 <= axis2_) and (axis2_ < len(input_shape))),   \
            "The argument axis2 is out of range (expected to be in range of [%d, %d], but got %d).\n"   \
            % (-(len(input_shape)), len(input_shape) - 1, axis2)


        assert  axis1_ != axis2_,   \
               "axis1 and axis2 cannot be the same axis." \
                "But received axis1 = %d, axis2 = %d\n"%(axis1, axis2)

    __check_input(input, offset, axis1, axis2)

    if in_dygraph_mode():
        return _C_ops.final_state_trace( x, offset, axis1, axis2 )

    if _in_legacy_dygraph():
        return _C_ops.trace(x, 'offset', offset, 'axis1', axis1, 'axis2', axis2)

    inputs = {'Input': [x]}
    attrs = {'offset': offset, 'axis1': axis1, 'axis2': axis2}
    helper = LayerHelper('trace', **locals())

    out = helper.create_variable_for_type_inference(dtype=x.dtype)

    helper.append_op(
        type='trace',
        inputs={'Input': [x]},
        attrs={'offset': offset,
               'axis1': axis1,
               'axis2': axis2},
        outputs={'Out': [out]})
    return out
```

> 概念解释：LayerHelper是一个用于创建op输出变量、向program中添加op的辅助工具类

- Python API 实现要点
    - 对输入参数进行合法性检查，即 `__check_input(input, offset, axis1, axis2)`
    - 添加动态图分支调用，即 `if in_dygraph_mode` 新动态图分支和 `if _in_legacy_dygraph` 旧动态图分支
    - 添加静态图分支调用，即dygraph分支后剩余的代码

- Python API 放置位置
    - 根据 API 自身属性，结合现有目录分类情况，放置导致对应子目录中的相应文件中
    - 可以参考 [飞桨官方 API 文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/index_cn.html) 中对各个子目录 **功能和包含的API** 的介绍

- Python API 文档
    - 参考示例格式进行添加，内容尽可能准确、翔实，详细规范请参考 [PaddlePaddle 文档](https://github.com/PaddlePaddle/docs/wiki)

## 5. 添加单元测试

单测包括对比前向Op不同设备(CPU、CUDA)的实现、对比反向OP不同设备(CPU、CUDA)的实现、反向Op的梯度测试。下面介绍介绍[`TraceOp`的单元测试](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/tests/unittests/test_trace_op.py)。

**注意：**

单测中的测试用例需要尽可能的覆盖Kernel中的所有分支。

### 5.1 前向 Operator 单测

Op单元测试继承自`OpTest`。各项具体的单元测试在`TestTraceOp`里完成。测试Operator，需要：

1. 在`setUp`函数定义输入、输出，以及相关的属性参数。
2. 生成随机的输入数据。
3. 在Python脚本中实现与前向operator相同的计算逻辑，得到输出值，与operator前向计算的输出进行对比。
4. 反向计算已经自动集成进测试框架，直接调用相应接口即可。


      ```python
      import paddle
      import unittest
      import numpy as np
      from op_test import OpTest


      class TestTraceOp(OpTest):
          def setUp(self):
              self.op_type = "trace"
              self.python_api = paddle.trace
              self.init_config()
              self.outputs = {'Out': self.target}

          def test_check_output(self):
              self.check_output(check_eager=True)

          def test_check_grad(self):
              self.check_grad(['Input'], 'Out', check_eager=True)

          def init_config(self):
              self.case = np.random.randn(20, 6).astype('float64')
              self.inputs = {'Input': self.case}
              self.attrs = {'offset': 0, 'axis1': 0, 'axis2': 1}
              self.target = np.trace(self.inputs['Input'])
      ```

    上面的代码首先导入依赖的包，下面是对`setUp`函数中操作的重要变量的详细解释：

    - `self.op_type = "trace" ` : 定义类型，与operator注册时注册的类型一致。
    - `self.python_api = paddle.trace` : 定义python api，与python调用接口一致。
    - `self.inputs` : 定义输入，类型为`numpy.array`，并初始化。
    - `self.outputs` : 定义输出，并在Python脚本中完成与operator同样的计算逻辑，返回Python端的计算结果。

### 5.2 反向 operator 单测

而反向测试中：

- `test_check_grad`中调用`check_grad`使用数值法检测梯度正确性和稳定性。
  - 第一个参数`['Input']` : 指定对输入变量`Input`做梯度检测。
  - 第二个参数`'Out'` : 指定前向网络最终的输出目标变量`Out`。
  - 第三个参数`check_eager` : `check_eager=True`表示开启新动态图（eager模式）单测，`check_eager`默认为`False`。

- 对于存在多个输入的反向Op测试，需要指定只计算部分输入梯度的case
  - 例如，`test_elementwise_sub_op.py`中的`test_check_grad_ingore_x`和`test_check_grad_ingore_y`分支用来测试只需要计算一个输入梯度的情况
  - 此处第三个参数max_relative_error：指定检测梯度时能容忍的最大错误值。

    ```python
    def test_check_grad_ingore_x(self):
        self.check_grad(
            ['Y'], 'Out', max_relative_error=0.005, no_grad_set=set("X"))

    def test_check_grad_ingore_y(self):
        self.check_grad(
            ['X'], 'Out', max_relative_error=0.005, no_grad_set=set('Y'))
    ```

其他有关单元测试添加的注意事项请参考 [《Op开发手册》](https://github.com/PaddlePaddle/Paddle/wiki/Operator-Development-Manual-Index) 及 [《Paddle单元测试规范》](https://github.com/PaddlePaddle/Paddle/wiki/PaddlePaddle-Unit-test-specification)。


### 5.3 编译和执行

`python/paddle/fluid/tests/unittests/` 目录下新增的 `test_*.py` 单元测试会被自动加入工程进行编译。

请注意，**运行单元测试测时需要编译整个工程**，并且编译时需要打开`WITH_TESTING`。

参考上述【3.3 编译测试】过程，编译成功后，在`build`目录下执行下面的命令来运行单元测试：

```bash
make test ARGS="-R test_trace_op -V"
```

或者执行:

```bash
ctest -R test_trace_op -V
```

**注意事项：**

- 注册Op时的类型名，需要和该Op的名字一样。即不允许在`A_op.cc`里面，注册`REGISTER_OPERATOR(B, ...)`等，这将会导致单元测试出错。

## 6. 其他编码要点

### 6.1 报错检查

实现Op时检查数据的合法性需要使用PADDLE_ENFORCE以及PADDLE_ENFORCE_EQ等宏定义，基本格式如下：

```
PADDLE_ENFORCE(表达式, 错误提示信息)
PADDLE_ENFORCE_EQ(比较对象A, 比较对象B, 错误提示信息)
```

如果表达式为真，或者比较对象A=B，则检查通过，否则会终止程序运行，向用户反馈相应的错误提示信息。
为了确保提示友好易懂，开发者需要注意其使用方法。

**总体原则：**
任何使用了PADDLE_ENFORCE与PADDLE_ENFORCE_XX检查的地方，必须有详略得当的备注解释！<font color="#FF0000">**错误提示信息不能为空！**</font>

报错提示信息书写建议：

1. [required] 哪里错了？为什么错了？

    - 例如：`ValueError: Mismatched label shape`

2. [optional] 期望的输入是什么样的？实际的输入是怎样的？

    - 例如：`Expected labels dimension=1. Received 4.`

3. [optional] 能否给出修改意见？

    - 例如：`Suggested Fix:If your classifier expects one-hot encoding label,check your n_classes argument to the estimatorand/or the shape of your label.Otherwise, check the shape of your label.`

更详细的报错检查规范介绍请参考 [《Paddle报错信息文案书写规范》](https://github.com/PaddlePaddle/Paddle/wiki/Paddle-Error-Message-Writing-Specification)。
