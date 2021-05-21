# How to write a new operator

<a name="Background"></a>
## Background

Here are the base types needed. For details, please refer to the design docs.

- `class OpProtoAndCheckerMaker`: Describes an Operator's input, output, attributes and description, mainly used to interface with Python API.
- `framework::OperatorBase`: Operator (Op)base class.
- `framework::OpKernel`: Base class for Op computation kernel.
- `framework::OperatorWithKernel`: Inherited from OperatorBase, describing an operator with computation kernels.


Operators can be categorized into two groups: operator with kernel(s) and operator without kernel(s). An operator with kernel(s) inherits from `OperatorWithKernel` while the one without kernel(s) inherits from `OperatorBase`. This tutorial focuses on implementing operators with kernels. In short, an operator includes the following information:


<table>
<thead>
<tr>
<th>Information</th>
<th> Where is it defined</th>
</tr>
</thead>
<tbody>
<tr>
<td>OpProtoMake definition </td>
<td> `.cc`files, Backward Op does not need an OpProtoMake interface. </td>
</tr>
<tr>
<td>Op definition  </td>
<td> `.cc` files</td>
</tr>
<tr>
<td>Kernel implementation  </td>
<td> The kernel methods shared between CPU and CUDA are defined in `.h` files. CPU-specific kernels live in `.cc` files, while CUDA-specific kernels are implemented in `.cu`files.</td>
</tr>
<tr>
<td>Registering the Op  </td>
<td> Ops are registered in `.cc` files; For Kernel registration, `.cc` files contain the CPU implementation, while `.cu` files contain the CUDA implementation.</td>
</tr>
</tbody>
</table>


New Operator implementations are added to the list [paddle/operators](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/fluid/operators), with file names in the format `*_op.h` (if applicable), `*_op.cc`, `*_op.cu` (if applicable).** The system will use the naming scheme to automatically build operators and their corresponding Python extensions.**


Let's take matrix multiplication operator, [MulOp](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/operators/mul_op.cc), as an example to introduce the writing of an Operator with Kernel.

<a name="Implementing C++ Types"></a>
## Implementing C++ Types

<a name="Defining ProtoMaker"></a>
### Defining ProtoMaker

Matrix Multiplication can be written as $Out = X * Y$, meaning that the operation consists of two inputs and one output.

First, define `ProtoMaker` to describe the Operator's input, output, and additional comments:

```cpp
class MulOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  MulOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "(Tensor), 2D tensor of size (M x K)");
    AddInput("Y", "(Tensor), 2D tensor of size (K x N)");
    AddOutput("Out", "(Tensor), 2D tensor of size (M x N)");
    AddComment(R"DOC(
Two Element Mul Operator.
The equation is: Out = X * Y
)DOC");
  }
};
```

[`MulOpMaker`](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/operators/mul_op.cc#L76-L127)is inherited from`framework::OpProtoAndCheckerMaker`, consisting of 2 variables in the constructor：

   - `framework::OpProto` stores Operator input and variable attribute, used for generating Python API interfaces.
   - `framework::OpAttrChecker` is used to validate variable attributes.

The constructor utilizes `AddInput` to add input parameter, `AddOutput` to add output parameter, and `AddComment` to add comments for the Op, so that the corresponding information will be added to `OpProto`.

The code above adds two inputs `X` and `Y` to `MulOp`, an output `Out`, and their corresponding descriptions. Names are given in accordance to Paddle's [naming convention](https://github.com/PaddlePaddle/FluidDoc/blob/release/1.2/doc/fluid/dev/name_convention.md).


An additional example [`ScaleOp`](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/operators/scale_op.cc#L38-L55) is implemented as follows:


```cpp
template <typename AttrType>
class ScaleOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  ScaleOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "(Tensor) Input tensor of scale operator.");
    AddOutput("Out", "(Tensor) Output tensor of scale operator.");
    AddComment(R"DOC(
Scale operator
$$Out = scale*X$$
)DOC");
    AddAttr<AttrType>("scale",
                      "(float, default 1.0)"
                      "The scaling factor of the scale operator.")
        .SetDefault(1.0);
  }
};
```

Note `AddAttr<AttrType>("scale", "...").SetDefault(1.0);` adds `scale`constant as an attribute, and sets the default value to 1.0.

<a name="Defining the GradProtoMaker class"></a>
### Defining the GradProtoMaker class

Each Op must have a corresponding GradProtoMaker. If GradProtoMaker corresponding to the forward Op is not customized, Fluid provides DefaultGradProtoMaker. The default registration will use all input and output, including Input, Output, Output@Grad and so on. Using unnecessary variables will cause waste of memory.
The following example defines ScaleOp's GradProtoMaker.

```cpp
class ScaleGradMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

  std::unique_ptr<framework::OpDesc> Apply() const override {
    auto *grad_op = new framework::OpDesc();
    grad_op->SetType("scale");
    grad_op->SetInput("X", OutputGrad("Out"));
    grad_op->SetOutput("Out", InputGrad("X"));
    grad_op->SetAttr("scale", GetAttr("scale"));
    return std::unique_ptr<framework::OpDesc>(grad_op);
  }
};
```
<a name="Defining Operator"></a>
### Defining Operator

The following code defines the interface for MulOp:

```cpp
class MulOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    //never use Input<Tensor> or Output<Tensor> if you want a to get a LoDTensor.
    auto dim0 = ctx.Input<LoDTensor>("X")->dims();
    auto dim1 = ctx.Input<LoDTensor>("Y")->dims();
    PADDLE_ENFORCE_EQ(dim0.size(), 2,
                      "input X(%s) should be a tensor with 2 dims, a matrix",
                      ctx.op_.Input("X"));
    PADDLE_ENFORCE_EQ(dim1.size(), 2,
                      "input Y(%s) should be a tensor with 2 dims, a matrix",
                      ctx.op_.Input("Y"));
    PADDLE_ENFORCE_EQ(
        dim0[1], dim1[0],
        "First matrix's width must be equal with second matrix's height.");
    ctx.Output<LoDTensor>("Out")->Resize({dim0[0], dim1[1]});
  }
};
```

[`MulOp`](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/operators/mul_op.cc#L24) is inherited from `OperatorWithKernel`. Its `public` member

```cpp
using framework::OperatorWithKernel::OperatorWithKernel;
```

expresses an operator constructor using base class `OperatorWithKernel`, alternatively written as

```cpp
MulOp(const std::string &type, const framework::VariableNameMap &inputs,
      const framework::VariableNameMap &outputs,
      const framework::AttributeMap &attrs)
  : OperatorWithKernel(type, inputs, outputs, attrs) {}
```

`InferShape` interface needs to be re-written.`InferShape` is a const method and cannot modify Op's member variables. Its constant member `const framework::InferShapeContext &ctx` can be used to extract input, output, and attributes. Its functions are

  - 1). validate and error out early: it checks input data dimensions and types.
  - 2). configures the tensor shape in the output.

Usually `OpProtoMaker` and `Op` definitions are written in `.cc` files, which also include the registration methods introduced later.

<a name="Defining OpKernel"></a>
### Defining OpKernel

`MulKernel` is derived from `framework::OpKernel`, which includes the following templates:

- `typename  DeviceContext` denotes device context type. When different devices, namely the CPU and the CUDA, share the same kernel, this template needs to be added. If they don't share kernels, this must not be added. An example of a non-sharing kernel is [`OnehotCrossEntropyOpKernel`](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/operators/cross_entropy_op.h#L43).

- `typename T` denotes data type, such as `float` or `double`.

`MulKernel` types need to rewrite the interface for `Compute`.

- `Compute` takes one input parameter: `const framework::ExecutionContext& context`.
- Compared with `InferShapeContext`, `ExecutionContext` includes device types, and can similarly extract input, output, and attribute variables.
- `Compute` function implements the computation logics of an `OpKernel`.

The input and output of Op can be obtained by `ExecutionContext::Input<T>()` and `ExecutionContext::Output<T>()` respectively.

**Note:** If the input/output variable type of op is `LoDTensor` (In Fluid, all Tensors are LoDTensor type by default), please write `ExecutionContext::Input<LoDTensor>()` and `ExecutionContext:: Output<LoDTensor>()`, do not write `ExecutionContext::Input<Tensor>()` and `ExecutionContext::Output<Tensor>()`. Because if the actual variable type is `SelectedRows`, the `Input<Tensor>()` and `Output<Tensor>()` methods will specialize the `SelectedRows` type to `Tensor`, causing a potential error.


`MulKernel`'s implementation of `Compute` is as follows:

```cpp
template <typename DeviceContext, typename T>
class MulKernel : public framework::OpKernel {
public:
void Compute(const framework::ExecutionContext& context) const override {
  auto* X = context.Input<LoDTensor>("X");
  auto* Y = context.Input<LoDTensor>("Y");
  auto* Z = context.Output<LoDTensor>("Out");
  Z->mutable_data<T>(context.GetPlace());
  auto& device_context = context.template device_context<DeviceContext>();
  math::matmul<DeviceContext, T>(*X, false, *Y, false, 1, Z, 0, device_context);
}
};
```

Note that **different devices (CPU, CUDA)share one Op definition; whether or not they share the same `OpKernel` depends on whether functions called by `Compute`can support both devices.**

`MulOp`'s CPU and CUDA share the same `Kernel`. A non-sharing  `OpKernel` example can be seen in [`OnehotCrossEntropyOpKernel`](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/operators/cross_entropy_op.cc).

To ease the writing of `OpKernel` compute, and for reusing code cross-device, [`Eigen-unsupported Tensor`](https://bitbucket.org) module is used to implement `Compute` interface. To learn about how the Eigen library is used in PaddlePaddle, please see [usage document](https://github.com/PaddlePaddle/FluidDoc/blob/release/1.2/doc/fluid/dev/use_eigen_cn.md).


This concludes the forward implementation of an operator. Next its operation and kernel need to be registered in a `.cc` file.

The definition of its corresponding backward operator, if applicable, is similar to that of an forward operator. **Note that a backward operator does not include a `ProtoMaker`**.


<a name="Registering Operator and OpKernel"></a>
### Registering Operator and OpKernel

- In `.cc` files, register forward and backward operator classes and the CPU kernel.

    ```cpp
    namespace ops = paddle::operators;
    REGISTER_OPERATOR(mul, ops::MulOp, ops::MulOpMaker,
                  paddle::framework::DefaultGradOpDescMaker<true>)
    REGISTER_OPERATOR(mul_grad, ops::MulGradOp)
    REGISTER_OP_CPU_KERNEL(mul, ops::MulKernel<paddle::platform::CPUDeviceContext, float>);
    REGISTER_OP_CPU_KERNEL(mul_grad,
                  ops::MulGradKernel<paddle::platform::CPUDeviceContext, float>);
    ```

    In that code block,

    - `REGISTER_OPERATOR` registers the `ops::MulOp` class, with the type named `mul`. Its `ProtoMaker` is `ops::MulOpMaker`. Register `ops::MulOpGrad` as type named `mul_grad`.
    - `REGISTER_OP_CPU_KERNEL` registers `ops::MulKernel` class and specializes template parameters as type `paddle::platform::CPUPlace` and `float`, and also registers `ops::MulGradKernel`.


- Registering CUDA Kernel in `.cu` files
    - Note that if CUDA Kernel is implemented using the `Eigen unsupported` module, then on top of `.cu`, a macro definition `#define EIGEN_USE_GPU` is needed, such as

    ```cpp
    // if use Eigen unsupported module before include head files
    #define EIGEN_USE_GPU

    namespace ops = paddle::operators;
    REGISTER_OP_CUDA_KERNEL(mul, ops::MulKernel<paddle::platform::CUDADeviceContext, float>);
    REGISTER_OP_CUDA_KERNEL(mul_grad,
                           ops::MulGradKernel<paddle::platform::CUDADeviceContext, float>);

    ```

<a name="Compilation"></a>
### Compilation

In folder `build/paddle/fluid/operators`, run the following commands to compile.

```
make mul_op
```

<a name="Python Binding"></a>
## Python Binding

The system will automatically bind the new op to Python and link it to a generated library.

<a name="Unit Tests"></a>
## Unit Tests

Unit tests for an operator include

1. comparing a forward operator's implementations on different devices (CPU, CUDA)

2. comparing a backward operator's implementation on different devices (CPU, CUDA)

3. a gradient test for the backward operator.

Here, we introduce the [unit tests for `MulOp`](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/tests/unittests/test_mul_op.py).


<a name="Unit Test for Forward Operators"></a>
### Unit Test for Forward Operators

The Op unit test is inherited from `OpTest`. More specific unit tests are done in `TestMulOp`. To test the Operator, you need to:

1. Define input, output, and related property parameters in the `setUp` function.
2. Generate random input data.
3. Implement the same calculation logic as the forward operator in the Python script to get the output, which is to be compared with the output of the forward operator calculation.
4. The backward calculation has been automatically integrated into the test framework and the corresponding interface can be called directly.

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


The code above first loads required packages. In addition, we have

- `self.op_type = "mul" ` defines the type that is identical to what the operator's registered type.
- `self.inputs` defines input, with type `numpy.array` and initializes it.
- `self.outputs` defines output and completes the same operator computation in the Python script, and returns its result from the Python script.

<a name="Unit test for backward operators"></a>
### Unit Test for Backward Operators

In the backward operator test:

- `check_grad` is called in `test_check_grad_normal` to use numerical methods to detect gradient correctness and stability.
- The first parameter `["X", "Y"]` : specifies gradient check for the input variables `X`, `Y`.
- The second parameter `"Out"` : specifies the final output target variable `Out` of the forward network.
- The third parameter `max_relative_error`: specifies the maximum error value that can be tolerated when checking gradients.
- The `test_check_grad_ingore_x` and `test_check_grad_ingore_y` branches are used to test cases where only one input gradient needs to be calculated.


<a name="Compiling and Running"></a>
### Compiling and Running


Any new unit testing file of the format `test_*.py`  added to the directory `python/paddle/fluid/tests/unittests/` is automatically added to the project to compile.

Note that **running unit tests requires compiling the entire project** and requires compiling with flag `WITH_TESTING` on i.e. `cmake paddle_dir -DWITH_TESTING=ON`.

After successfully compiling the project, run the following command to run unit tests:

```bash
make test ARGS="-R test_mul_op -V"
```

Or,

```bash
ctest -R test_mul_op
```


<a name="Remarks"></a>
## Remarks

- The type with which an operator is registered needs to be identical to the Op's name. Registering `REGISTER_OPERATOR(B, ...)` in `A_op.cc` will cause unit testing failures.
- If the operator does not implement a CUDA kernel, please refrain from creating an empty `*_op.cu` file, or else unit tests will fail.
- If multiple operators rely on some shared methods, a file NOT named `*_op.*` can be created to store them, such as `gather.h`.




<a name="PADDLE_ENFORCE Usage Note"></a>
### PADDLE_ENFORCE Usage Note

To check the validity of data when implementing Op, you need to use macro definitions such as PADDLE_ENFORCE and PADDLE_ENFORCE_EQ. The basic format is as follows:

```
PADDLE_ENFORCE (expression, error message)
PADDLE_ENFORCE_EQ (comparison object A, comparison object B, error message)
```

If the expression is true, or the comparison object A=B, the check will be passed, otherwise the program will be terminated and the corresponding error message will be fed back to the user.
In order to ensure that the feedbacks are user-friendly and easy to understand, developers need to pay attention to how to use them.


<a name="General Principles"></a>
#### General Principles

Any place where PADDLE_ENFORCE and PADDLE_ENFORCE_EQ are used must have a properly detailed explanation of the comments! **Error message** can't be empty!


<a name="Error Message Standard"></a>
#### Error Message Standard

1. [required] Where does it go wrong? Why is it wrong?

    - For example: `ValueError: Mismatched label shape`

2. [optional] What is the expected input? What is the actual input?

    - For example: `Expected labels dimension=1. Received 4.`

3. [optional] Can you come up with a suggestion?

    - For example: `Suggested Fix: If your classifier expects one-hot encoding label, check your n_classes argument to the estimatorand/or the shape of your label.Otherwise, check the shape of your label.`

If it is not necessary or concise description is enough to clearly express the above points, just write based on actual needs.


<a name="Typical Problems"></a>
#### Typical Problems


1.No error message exists or error message is too short to provide effective notification to the user.

  Problem example 1: Absent message
  ```
  PADDLE_ENFORCE(ctx->HasInput("X"), "");
  ```
  Problem example 2: The prompt message is too short
  ```
  PADDLE_ENFORCE(i != nullptr, "i must be set"); // What is i?
  ```

2.Using developer-defined variable abbreviations in error messages is not easy to understand.

  Example of the problem:
  ```
  PADDLE_ENFORCE(forward_pd != nullptr,
  "Fail to find eltwise_fwd_pd in device context"); //eltwise_fwd_pduser may not be understood
  ```

3.The OP internally calls the illegal interface: If Op appears inside Output = ShareDataWith(Input)
   Example of the problem:
   ```cpp
   auto *out = ctx.Output<framework::LoDTensor>("Out");
   auto *in = ctx.Input<framework::LoDTensor>("X");
   out->ShareDataWith(*in);
   ```

   If there is Output = ShareDataWith(Input) inside Op, it will equivalently indicate a hidden edge in the operator graph, which connects Input and Output. This edge cannot be expressed in graph analysis, causing error based on graph optimization.

4.Performance of OP implementation. It called eigen's broadcast, chop and other operations, the performance will be over several times worse than the handwritten cuda kernel. At this point, the implementation of cpu can reuse eigen, and the gpu implementation can implement cuda kernel.



<a name="Special instructions for OP InferShape check message"></a>
#### Special Instructions for OP InferShape Check Message

- Check input and output variables, please follow the following format
`Input(variable name) of OP name operator should not be null.`

  The correct example:
  ```
  PADDLE_ENFORCE(ctx->HasInput("Input"),
            "Input(Input) of LSTMP operator should not be null.");
  ```

- Backward Op input and output check, to write the name of the backward Op

  The correct example:
  ```
  PADDLE_ENFORCE(ctx->HasInput("X"),
              "Input(X) of LoDResetGrad opreator should not be null.");
  ```
