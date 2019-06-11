# 如何写新的C++ OP

## 概念简介

简单介绍需要用到基类，详细介绍请参考[设计文档](https://github.com/PaddlePaddle/FluidDoc/blob/develop/doc/fluid/design/motivation/refactorization.md#operatoropwithkernelopkernel)。

- `framework::OperatorBase`: Operator(简写，Op)基类。
- `framework::OpKernel`: Op计算函数的基类，称作Kernel。
- `framework::OperatorWithKernel`：继承自OperatorBase，Op有计算函数，称作有Kernel。
- `framework::OpProtoAndCheckerMaker`：描述该Op的输入、输出、属性、注释，主要用于Python API接口生成。

根据是否包含Kernel，可以将Op分为两种：包含Kernel的Op和不包含kernel的Op：

- 包含Kernel的Op继承自`OperatorWithKernel`，这类Op的功能实现与输入的数据类型、数据布局、数据所在的设备以及Op实现所调用第三方库等有关。比如ConvOp，如果使用CPU计算，一般通过调用mkl库中的矩阵乘操作实现，如果使用GPU计算，一般通过调用cublas库中的矩阵乘操作实现，或者直接调用cudnn库中的卷积操作。
- 不包含Kernel的Op继承自`OperatorBase`，因为这类Op的功能实现与设备以及输入的数据不相关。比如WhileOp、IfElseOp等。

本教程主要介绍带Kernel的Op如何写，简单总结Op需要包含的内容如下：

<table>
<thead>
<tr>
<th>内容</th>
<th>定义位置</th>
</tr>
</thead>
<tbody>
<tr>
<td>OpProtoMake定义 </td>
<td>.cc 文件 </td>
</tr>
<tr>
<td>Op定义 </td>
<td> .cc 文件</td>
</tr>
<tr>
<td>Kernel实现 </td>
<td> CPU、CUDA共享Kernel实现在.h 文件中，否则，CPU 实现在.cc 文件中，CUDA 实现在.cu 文件中。</td>
</tr>
<tr>
<td>注册Op </td>
<td> Op注册实现在.cc 文件；Kernel注册CPU实现在.cc 文件中，CUDA实现在.cu 文件中</td>
</tr>
</tbody>
</table>

实现新的op都添加至目录[paddle/fluid/operators](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/fluid/operators)下，文件命名以`*_op.h`（如有）、`*_op.cc` 、`*_op.cu`（如有）结尾。**系统会根据文件名自动构建op和其对应的Python扩展。**

下面以矩阵乘操作，即[MulOp](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/operators/mul_op.cc)为例来介绍如何写带Kernel的Operator。

## 实现C++类
### 定义ProtoMaker类

矩阵乘法的公式：$Out = X * Y$, 可见该计算由两个输入，一个输出组成。

首先定义`ProtoMaker`来描述该Op的输入、输出，并添加注释：

```cpp
class MulOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), The first input tensor of mul op.");
    AddInput("Y", "(Tensor), The second input tensor of mul op.");
    AddOutput("Out", "(Tensor), The output tensor of mul op.");
    AddAttr<int>(
        "x_num_col_dims",
        R"DOC((int, default 1), The mul_op can take tensors with more than two
              dimensions as its inputs. If the input $X$ is a tensor with more
              than two dimensions, $X$ will be flattened into a two-dimensional
              matrix first. The flattening rule is: the first `num_col_dims`
              will be flattened to form the first dimension of the final matrix
              (the height of the matrix), and the rest `rank(X) - num_col_dims`
              dimensions are flattened to form the second dimension of the final
              matrix (the width of the matrix). As a result, height of the
              flattened matrix is equal to the product of $X$'s first
              `x_num_col_dims` dimensions' sizes, and width of the flattened
              matrix is equal to the product of $X$'s last `rank(x) - num_col_dims`
              dimensions' size. For example, suppose $X$ is a 6-dimensional
              tensor with the shape [2, 3, 4, 5, 6], and `x_num_col_dims` = 3.
              Thus, the flattened matrix will have a shape [2 x 3 x 4, 5 x 6] =
              [24, 30].
        )DOC")
        .SetDefault(1)
        .EqualGreaterThan(1);
    AddAttr<int>(
        "y_num_col_dims",
        R"DOC((int, default 1), The mul_op can take tensors with more than two,
              dimensions as its inputs. If the input $Y$ is a tensor with more
              than two dimensions, $Y$ will be flattened into a two-dimensional
              matrix first. The attribute `y_num_col_dims` determines how $Y$ is
              flattened. See comments of `x_num_col_dims` for more details.
        )DOC")
        .SetDefault(1)
        .EqualGreaterThan(1);
    AddComment(R"DOC(
Mul Operator.

This operator is used to perform matrix multiplication for input $X$ and $Y$.

The equation is:

$$Out = X * Y$$

Both the input $X$ and $Y$ can carry the LoD (Level of Details) information,
or not. But the output only shares the LoD information with input $X$.

)DOC");
  }
};
```

[`MulOpMaker`](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/operators/mul_op.cc)继承自`framework::OpProtoAndCheckerMaker`。

开发者通过覆盖`framework::OpProtoAndCheckerMaker`中的`Make`函数来定义Op所对应的Proto，通过`AddInput`添加输入参数，通过`AddOutput`添加输出参数，通过`AddAttr`添加属性参数，通过`AddComment`添加Op的注释。这些函数会将对应内容添加到`OpProto`中。

上面的代码在`MulOp`中添加两个输入`X`和`Y`，添加了一个输出`Out`，并解释了各自含义，命名请遵守[命名规范](https://github.com/PaddlePaddle/FluidDoc/blob/release/1.2/doc/fluid/dev/name_convention.md)。

### 定义GradProtoMaker类
通常情况下，每个Op的会有一个对应的`GradProtoMaker`，为方便代码编写，fluid提供了默认的`GradProtoMaker`，即：`DefaultGradProtoMaker`。`DefaultGradProtoMaker`会使用前向Op的全部输入(`Input`)输出(`Output`)以及输出变量所对应的梯度（`Output@Grad`）作为反向Op的输入，将前向Op的输入变量所对应的的梯度（`Input@Grad`）作为输出。

**注意:**
不要将反向Op不会用到的变量放到反向Op的输入列表中，这样会导致这些不会被反向Op用到的变量的空间不能够及时回收，进而有可能导致用到该Op的模型可以设置的batch_size较低。
比如`relu`操作的前向操作为：`out.device(d) = x.cwiseMax(static_cast<T>(0));`反向操作为：`dx.device(d) = dout * (out > static_cast<T>(0)).template cast<T>();`。显然，反向操作中只是用到了`out`、`dout`、`dx`，没有用到`x`。


下面示例定义了`MulOp`的GradProtoMaker。

```cpp
class MulOpGradMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    std::unique_ptr<framework::OpDesc> retv(new framework::OpDesc());
    retv->SetType("mul_grad");
    retv->SetInput("X", Input("X"));
    retv->SetInput("Y", Input("Y"));
    retv->SetInput(framework::GradVarName("Out"), OutputGrad("Out"));
    retv->SetOutput(framework::GradVarName("X"), InputGrad("X"));
    retv->SetOutput(framework::GradVarName("Y"), InputGrad("Y"));
    retv->SetAttrMap(Attrs());
    return retv;
  }
};
```

**注意：**

- 有些Op的前向逻辑和反向逻辑是一样的，比如[`ScaleOp`](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/operators/scale_op.cc).这种情况下，前向Op和反向Op的Kernel可以为同一个。
- 有些前向Op所对应的反向Op可能有多个，比如[`SumOp`](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/operators/sum_op.cc)，这种情况下，`GradMaker`需要继承`framework::GradOpDescMakerBase`。
- 有些Op的反向对应另一个Op的前向，比如[`SplitOp`](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/operators/split_op.h)，这种情况下，[`SplitGradMaker`](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/operators/split_op.h#L52)中定义的`SplitOp`反向Op的Type就是`concat`，

### 定义Operator类

下面实现了MulOp的定义：

```cpp
class MulOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) of MulOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Y"), "Input(Y) of MulOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of MulOp should not be null.");

    auto x_dims = ctx->GetInputDim("X");
    auto y_dims = ctx->GetInputDim("Y");

    int x_num_col_dims = ctx->Attrs().Get<int>("x_num_col_dims");
    int y_num_col_dims = ctx->Attrs().Get<int>("y_num_col_dims");

    VLOG(3) << "mul operator x.shape=" << x_dims << " y.shape=" << y_dims
            << " x_num_col_dims=" << x_num_col_dims
            << " y_num_col_dims=" << y_num_col_dims;

    PADDLE_ENFORCE_GT(
        x_dims.size(), x_num_col_dims,
        "The input tensor X's rank of MulOp should be larger than "
        "x_num_col_dims.");
    PADDLE_ENFORCE_GT(
        y_dims.size(), y_num_col_dims,
        "The input tensor Y's rank of MulOp should be larger than "
        "y_num_col_dims: %ld vs %ld",
        y_dims.size(), y_num_col_dims);

    auto x_mat_dims = framework::flatten_to_2d(x_dims, x_num_col_dims);
    auto y_mat_dims = framework::flatten_to_2d(y_dims, y_num_col_dims);

    PADDLE_ENFORCE_EQ(x_mat_dims[1], y_mat_dims[0],
                      "First matrix's width must be equal with second matrix's "
                      "height. %s, %s",
                      x_mat_dims[1], y_mat_dims[0]);
    std::vector<int64_t> output_dims;
    output_dims.reserve(
        static_cast<size_t>(x_num_col_dims + y_dims.size() - y_num_col_dims));

    for (int i = 0; i < x_num_col_dims; ++i) {
      output_dims.push_back(x_dims[i]);
    }

    for (int i = y_num_col_dims; i < y_dims.size(); ++i) {
      output_dims.push_back(y_dims[i]);
    }

    ctx->SetOutputDim("Out", framework::make_ddim(output_dims));
    ctx->ShareLoD("X", /*->*/ "Out");
  }
};
```

[`MulOp`](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/operators/mul_op.cc#L22)继承自`OperatorWithKernel`。`public`成员：

```cpp
using framework::OperatorWithKernel::OperatorWithKernel;
```

这句表示使用基类`OperatorWithKernel`的构造函数，也可写成：

```cpp
MulOp(const std::string &type, const framework::VariableNameMap &inputs,
      const framework::VariableNameMap &outputs,
      const framework::AttributeMap &attrs)
  : OperatorWithKernel(type, inputs, outputs, attrs) {}
```

还需要重写`InferShape`接口。`InferShape`为const函数，不能修改Op的成员变量，参数为`framework::InferShapeContext* ctx`，通过该参数可获取到输入输出以及属性。它的功能是：

  - 做检查， 尽早报错：检查输入数据维度、类型等是否合法。
  - 设置输出Tensor的形状以及LoD信息。

通常`OpProtoMaker`和`Op`类的定义写在`.cc`文件中，和下面将要介绍的注册函数一起放在`.cc`中

### InferShape区分 compile time 和 run time
在我们的静态图网络中，`InferShape`操作在[编译时(compile time)和运行时(run time)](https://github.com/PaddlePaddle/FluidDoc/blob/release/1.2/doc/fluid/getstarted/Developer's_Guide_to_Paddle_Fluid.md#%E8%AE%A9%E6%88%91%E4%BB%AC%E5%9C%A8fluid%E7%A8%8B%E5%BA%8F%E5%AE%9E%E4%BE%8B%E4%B8%AD%E5%8C%BA%E5%88%86%E7%BC%96%E8%AF%91%E6%97%B6%E5%92%8C%E8%BF%90%E8%A1%8C%E6%97%B6)都会被调用，在compile time时，由于真实的维度未知，框架内部用-1来表示，在run time时，用实际的维度表示，因此维度的值在compile time和 run time时可能不一致，如果存在维度的判断和运算操作，InferShape就需要区分compile time 和 run time。

以下两种情况需要区分compile time和 run time。

**1.检查**

如以下代码：
```cpp
auto x_dim = ctx->GetInputDim("X");
int i = xxx;
PADDLE_ENFORCE_GT( x_dim[i] , 10)
```

在compile time的时候，x_dim[i]可能等于-1，导致这个PADDLE_ENFORCE_GT报错退出。

如果用了以下paddle中定义的宏进行判断：
```cpp
PADDLE_ENFORCE_EQ ( x_dim[i] , 10)
PADDLE_ENFORCE_NE ( x_dim[i] , 10)
PADDLE_ENFORCE_GT ( x_dim[i] , 10)
PADDLE_ENFORCE_GE ( x_dim[i] , 10)
PADDLE_ENFORCE_LT ( x_dim[i] , 10)
PADDLE_ENFORCE_LE ( x_dim[i] , 10)
```
都需要区分compile time和run time

**2. 运算**

如以下代码:
```cpp
auto x_dim = ctx->GetInputDim("X");
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

**处理的标准**：
- 检查： compile time的时候不判断维度等于-1的情况，但在runtime的时候检查
- 运算： -1和其他数做任何运算都要等于-1

**参考代码**
1. 判断的实现方法可以参考cross_entropy_op.cc，cross_entropy_op 要求X和labels的两个输入，除了最后一维以外，其他的维度完全一致

```cpp
    bool contain_unknown_dim = framework::contain_unknown_dim(x_dims) ||
                               framework::contain_unknown_dim(label_dims);
    bool check = ctx->IsRuntime() || !contain_unknown_dim;
    if (check) {
      PADDLE_ENFORCE_EQ(framework::slice_ddim(x_dims, 0, rank - 1),
                        framework::slice_ddim(label_dims, 0, rank - 1),
                        "Input(X) and Input(Label) shall have the same shape "
                        "except the last dimension.");
    }
```

2. 运算的实现可以参考concat_op.cc，concat在InferShape判断时，除了进行concat轴之外，其他的维度完全一致；在生成output的维度时，把concat轴的维度求和，其他的维度和输入保持一致。

```cpp
    auto out_dims = ins[0];
    size_t in_zero_dims_size = out_dims.size();
    for (size_t i = 1; i < n; i++) {
      for (size_t j = 0; j < in_zero_dims_size; j++) {
        if (j == axis) {
          if (ctx->IsRuntime()) {
            out_dims[axis] += ins[i][j];
          } else {
            if (ins[i][j] == -1) {
              out_dims[axis] = -1;
            } else {
              out_dims[axis] += ins[i][j];
            }
          }
        } else {
          bool check_shape =
              ctx->IsRuntime() || (out_dims[j] > 0 && ins[i][j] > 0);
          if (check_shape) {
            // check all shape in run time
            PADDLE_ENFORCE_EQ(out_dims[j], ins[i][j],
                              "Input tensors should have the same "
                              "elements except the specify axis.");
          }
        }
      }
    }
```



### 定义OpKernel类

`MulKernel`继承自`framework::OpKernel`，带有下面两个模板参数:

- `typename DeviceContext`: 表示设备类型。不同设备(CPU、CUDA)共享同一个Kernel时，需加该模板参数；不共享则不加，一个不共享的例子是[`SGDOpKernel`](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/operators/optimizers/sgd_op.h)。

- `typename T` : 表示数据类型，如`float`, `double`, `int16`等。

需要为`MulKernel`类重写`Compute`接口。

- `Compute`接受一个输入参数：`const framework::ExecutionContext& context`。

- 与`InferShapeContext`相比，`ExecutionContext`增加了设备类型，同样可获取到输入输出和属性参数。

- `Compute`函数里实现`OpKernel`的具体计算逻辑。

Op的输入和输出可分别通过`ExecutionContext::Input<T>()`和`ExecutionContext::Output<T>()`获得。

**注意：** 若op的输入/输出的变量类型是`LoDTensor`（fluid默认所有的`Tensor`默认都是`LoDTensor`类型），请写成`ExecutionContext::Input<LoDTensor>()`和`ExecutionContext::Output<LoDTensor>()`，不要写`ExecutionContext::Input<Tensor>()`和`ExecutionContext::Output<Tensor>()`。因为若实际的变量类型为`SelectedRows`，`Input<Tensor>()`和`Output<Tensor>()`方法会将`SelectedRows`类型特化为`Tensor`，导致潜在的错误。

下面是 `MulKernel` `Compute`的实现：

```cpp
template <typename DeviceContext, typename T>
class MulKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* x = context.Input<Tensor>("X");
    const Tensor* y = context.Input<Tensor>("Y");
    Tensor* z = context.Output<Tensor>("Out");
    const Tensor x_matrix =
        x->dims().size() > 2
            ? framework::ReshapeToMatrix(
                  *x, context.template Attr<int>("x_num_col_dims"))
            : *x;
    const Tensor y_matrix =
        y->dims().size() > 2
            ? framework::ReshapeToMatrix(
                  *y, context.template Attr<int>("y_num_col_dims"))
            : *y;

    z->mutable_data<T>(context.GetPlace());
    auto z_dim = z->dims();
    if (z_dim.size() != 2) {
      z->Resize({x_matrix.dims()[0], y_matrix.dims()[1]});
    }

    auto blas = math::GetBlas<DeviceContext, T>(context);

    blas.MatMul(x_matrix, y_matrix, z);
    if (z_dim.size() != 2) {
      z->Resize(z_dim);
    }
  }
};
```

需要注意：**不同设备(CPU、CUDA)共享一个Op定义，是否则共享同一个`OpKernel`，取决于`Compute`调用的函数是否支持不同设备。**

`MulOp`的CPU、CUDA实现共享同一个`Kernel`。`OpKernel`不共享的例子可以参考：[`SGDOpKernel`](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/operators/optimizers/sgd_op.h)。

为了使`OpKernel`的计算过程书写更加简单，并且CPU、CUDA的代码可以复用，我们通常借助 Eigen unsupported Tensor模块来实现`Compute`接口。关于在PaddlePaddle中如何使用Eigen库，请参考[使用文档](https://github.com/PaddlePaddle/FluidDoc/blob/release/1.2/doc/fluid/dev/use_eigen_cn.md)。

到此，前向Op实现完成。接下来，需要在`.cc`文件中注册该op和kernel。
反向Op类的定义，反向OpKernel的定义与前向Op类似，这里不再赘述。

### 注册Operator

- 在`.cc`文件中注册前向、反向Op类，注册CPU Kernel。

    ```cpp
    namespace ops = paddle::operators;
    REGISTER_OPERATOR(mul, ops::MulOp, ops::MulOpMaker,
                  ops::MulOpGradMaker)
    REGISTER_OPERATOR(mul_grad, ops::MulGradOp)
    REGISTER_OP_CPU_KERNEL(mul, 
                  ops::MulKernel<paddle::platform::CPUDeviceContext, float>,
                  ops::MulKernel<paddle::platform::CPUDeviceContext, double>);
    REGISTER_OP_CPU_KERNEL(mul_grad,
                  ops::MulGradKernel<paddle::platform::CPUDeviceContext, float>,
                  ops::MulGradKernel<paddle::platform::CPUDeviceContext, double>);
    ```

    在上面的代码中：

	   - `REGISTER_OPERATOR` ： 注册`ops::MulOp`类，类型名为`mul`，该类的`ProtoMaker`为`ops::MulOpMaker`，注册`ops::MulOpGrad`，类型名为`mul_grad`。

	   - `REGISTER_OP_CPU_KERNEL` ：注册`ops::MulKernel`类，并特化模板参数为`paddle::platform::CPUPlace`和`float`类型，同理，注册`ops::MulGradKernel`类。


- 在 `.cu`文件中注册CUDA Kernel。
    - 请注意，如果CUDA Kernel的实现基于Eigen unsupported模块，那么在 `.cu`的开始请加上宏定义 `#define EIGEN_USE_GPU`，代码示例如下：


    ```cpp
    // if use Eigen unsupported module before include head files
    #define EIGEN_USE_GPU

    namespace ops = paddle::operators;
    REGISTER_OP_CUDA_KERNEL(mul, 
                            ops::MulKernel<paddle::platform::CUDADeviceContext, float>,
                            ops::MulKernel<paddle::platform::CUDADeviceContext, double>);
    REGISTER_OP_CUDA_KERNEL(mul_grad,
                            ops::MulGradKernel<paddle::platform::CUDADeviceContext, float>,
                            ops::MulGradKernel<paddle::platform::CUDADeviceContext, double>);
    ```

**注意：**

在运行Op时，框架系统会根据输入数据所在的设备、输入数据的类型等信息自动的选择合适的OpKernel，比如输入的数据是在GPU上，并且为`float`类型，框架系统会选择由`REGISTER_OP_CUDA_KERNEL`注册的`ops::MulKernel<paddle::platform::CUDADeviceContext, float>`。如果用户希望指定运行时可被调用的OpKernel，用户需要覆盖`framework::OperatorWithKernel`中的`GetExpectedKernelType`函数，比如`ConvOp`会根据属性`use_cudnn`为`false`还是为`true`决定是否调用cudnn库中提供的conv操作。

```
framework::OpKernelType ConvOp::GetExpectedKernelType(
    const framework::ExecutionContext& ctx) const {
  int customized_type_value =
      framework::OpKernelType::kDefaultCustomizedTypeValue;
  framework::LibraryType library{framework::LibraryType::kPlain};
  auto input_data_type = ctx.Input<Tensor>("Input")->type();
  std::string data_format = ctx.Attr<std::string>("data_format");
  framework::DataLayout layout = framework::StringToDataLayout(data_format);
#ifdef PADDLE_WITH_CUDA
  if (ctx.Attr<bool>("use_cudnn")) {
    library = framework::LibraryType::kCUDNN;
  }
#endif
  auto type = framework::OpKernelType(input_data_type, ctx.GetPlace(), layout,
                                      library, customized_type_value);
  return type;
}
```

### 编译

运行下面命令可以进行编译：

```
make mul_op
```

## 绑定Python

系统会对新增的op自动绑定Python，并链接到生成的lib库中。

### 使用mul操作在Python端构建Layer

在Python端，`mul`操作用于构建FC层，即：

$$Out = Act({X*W + b})$$

具体实现方式可参考[FC层的实现代码](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/layers/nn.py#L205)。

## 实现单元测试

单测包括对比前向Op不同设备(CPU、CUDA)的实现、对比反向OP不同设备(CPU、CUDA)的实现、反向Op的梯度测试。下面介绍介绍[`MulOp`的单元测试](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/tests/unittests/test_mul_op.py)。

**注意：**

单测中的测试用例需要尽可能的覆盖Op中的所有分支。

### 前向Operator单测

Op单元测试继承自`OpTest`。各项具体的单元测试在`TestMulOp`里完成。测试Operator，需要：

1. 在`setUp`函数定义输入、输出，以及相关的属性参数。
2. 生成随机的输入数据。
3. 在Python脚本中实现与前向operator相同的计算逻辑，得到输出值，与operator前向计算的输出进行对比。
4. 反向计算已经自动集成进测试框架，直接调用相应接口即可。


	  ```python
	  import unittest
	  import numpy as np
	  from op_test import OpTest


	  class TestMulOp(OpTest):
	      def setUp(self):
	          self.op_type = "mul"
	          self.inputs = {
	              'X': np.random.random((32, 84)).astype("float32"),
	              'Y': np.random.random((84, 100)).astype("float32")
	          }
	          self.outputs = {'Out': np.dot(self.inputs['X'], self.inputs['Y'])}

	      def test_check_output(self):
	          self.check_output()

	      def test_check_grad_normal(self):
	          self.check_grad(['X', 'Y'], 'Out', max_relative_error=0.5)

	      def test_check_grad_ingore_x(self):
	          self.check_grad(
	              ['Y'], 'Out', max_relative_error=0.5, no_grad_set=set("X"))

	      def test_check_grad_ingore_y(self):
	          self.check_grad(
	              ['X'], 'Out', max_relative_error=0.5, no_grad_set=set('Y'))
	  ```

	上面的代码首先导入依赖的包，下面是对`setUp`函数中操作的重要变量的详细解释：

	- `self.op_type = "mul" ` : 定义类型，与operator注册时注册的类型一致。
	- `self.inputs` : 定义输入，类型为`numpy.array`，并初始化。
	- `self.outputs` : 定义输出，并在Python脚本中完成与operator同样的计算逻辑，返回Python端的计算结果。

### 反向operator单测

而反向测试中：

- `test_check_grad_normal`中调用`check_grad`使用数值法检测梯度正确性和稳定性。
  - 第一个参数`["X", "Y"]` : 指定对输入变量`X`、`Y`做梯度检测。
  - 第二个参数`"Out"` : 指定前向网络最终的输出目标变量`Out`。
  - 第三个参数`max_relative_error`：指定检测梯度时能容忍的最大错误值。

- `test_check_grad_ingore_x`和`test_check_grad_ingore_y`分支用来测试只需要计算一个输入梯度的情况。


### 编译和执行

`python/paddle/fluid/tests/unittests/` 目录下新增的 `test_*.py` 单元测试会被自动加入工程进行编译。

请注意，**不同于Op的编译测试，运行单元测试测时需要编译整个工程**，并且编译时需要打开`WITH_TESTING`, 即`cmake -DWITH_TESTING=ON ..`。编译成功后，执行下面的命令来运行单元测试：

```bash
make test ARGS="-R test_mul_op -V"
```

或者:

```bash
ctest -R test_mul_op
```

## 注意事项

- 注册Op时的类型名，需要和该Op的名字一样。即不允许在`A_op.cc`里面，注册`REGISTER_OPERATOR(B, ...)`等，这将会导致单元测试出错。
- 如果Op没有实现CUDA Kernel，请不要创建空的`*_op.cu`，这将会导致单元测试出错。
- 如果多个Op依赖一些共用的函数，可以创建非`*_op.*`格式的文件来存放，如`gather.h`文件。

### PADDLE_ENFORCE使用注意

实现Op时检查数据的合法性需要使用PADDLE_ENFORCE以及PADDLE_ENFORCE_EQ等宏定义，基本格式如下：

```
PADDLE_ENFORCE(表达式, 错误提示信息)
PADDLE_ENFORCE_EQ(比较对象A, 比较对象B, 错误提示信息)
```

如果表达式为真，或者比较对象A=B，则检查通过，否则会终止程序运行，向用户反馈相应的错误提示信息。
为了确保提示友好易懂，开发者需要注意其使用方法。

#### 总体原则

任何使用了PADDLE_ENFORCE与PADDLE_ENFORCE_XX检查的地方，必须有详略得当的备注解释！<font color="#FF0000">**错误提示信息不能为空！**</font>

#### 提示信息书写标准

1. [required] 哪里错了？为什么错了？

    - 例如：`ValueError: Mismatched label shape`

2. [optional] 期望的输入是什么样的？实际的输入是怎样的？

    - 例如：`Expected labels dimension=1. Received 4.`

3. [optional] 能否给出修改意见？

    - 例如：`Suggested Fix:If your classifier expects one-hot encoding label,check your n_classes argument to the estimatorand/or the shape of your label.Otherwise, check the shape of your label.`

如果并非必要或者简洁的描述即可表达清楚以上要点，根据情况书写亦可。

#### FAQ 典型问题

1. 无报错信息或报错信息过于简单，不能给用户提供有效的提示！

	问题示例1 ：未写提示信息
	```
	PADDLE_ENFORCE(ctx->HasInput("X"), "");
	```
	问题示例2 ：提示信息过于简单
	```
	PADDLE_ENFORCE(i != nullptr, "i must be set"); // i是什么？
	```

2. 在报错信息中使用开发人员定义的变量缩写，不易理解！

	问题示例：
	```
	PADDLE_ENFORCE(forward_pd != nullptr,
	                    "Fail to find eltwise_fwd_pd in device context");  //eltwise_fwd_pd用户可能看不懂
	```

3. OP内部调用非法接口：Op内部如果出现Output = ShareDataWith(Input)
	问题示例：
	```cpp
	auto *out = ctx.Output<framework::LoDTensor>("Out");
	auto *in = ctx.Input<framework::LoDTensor>("X");
	out->ShareDataWith(*in);
	```
	Op内部如果出现Output = ShareDataWith(Input)，相当于operator图的中有一条隐藏边，连接了Input和Output，这条边无法在图分析中表达，引发基于图优化的错误。

4. OP实现的性能实践
	调用了eigen的broadcast, chop等操作，性能会比手写cuda kernel差几倍以上。此时cpu的实现可以复用eigen，gpu实现可以实现cuda kernel.


#### OP InferShape检查提示信息特别说明

- 检查输入输出变量，请统一遵循以下格式
`Input(变量名) of OP名 operator should not be null.`

	正确示例：
	```
	PADDLE_ENFORCE(ctx->HasInput("Input"),
	                        "Input(Input) of LSTMP operator should not be null.");
	```

- 反向Op的输入输出检查，要写明反向Op的名字

	正确示例：
	```
	PADDLE_ENFORCE(ctx->HasInput("X"),
	                        "Input(X) of LoDResetGrad opreator should not be null.");
	```
