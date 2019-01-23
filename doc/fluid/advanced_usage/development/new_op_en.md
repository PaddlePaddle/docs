# How to code a new operator

 - [Conceptual introduction](#Conceptual introduction)
 - [Implement C++ class](#implement c class)
   - [Define ProtoMaker class](#definition protomaker class)
   - [Define the Operator class](#Define the operator class)
   - [Define OpKernel class](#definition opkernel class)
   - [Register Operator](#Register operator)
   - [Compile](#Compile)
 - [Bind Python](#binding python)
 - [Implement unit test](#implement unit test)
   - [Forward Operator Single Test](# forward operator single test)
   - [Reverse Operator Single Test](#reverse operator single test)
   - [Compile and Execute](#Compile and Execute)
 - [Precautions](#Notes)


## Concept Introduction

A brief introduction requires the use of a base class. Please refer to the design documentation for details.

- `framework::OperatorBase`: Operator (shorthand, Op) base class.
- `framework::OpKernel`: The base class for the Op calculation function, called Kernel.
- `framework::OperatorWithKernel`: Inherited from OperatorBase, Op has a calculation function called Kernel.
- `class OpProtoAndCheckerMaker`: describes the input, output, properties, and comments of the Op, mainly used for Python API interface generation.

According to whether including the kernel or not, Op can be divided into two types: Op containing Kernel and Op not containing kernel. The definition of Op is inherited from `OperatorWithKernel`, and the latter is inherited from `OperatorBase`. This tutorial mainly introduces how to write Op with Kernel. Simply summarize the contents that Op needs to include as follows:

<table>
<thead>
<tr>
<th>Content</th>
<th>Define a location</th>
</tr>
</thead>
<tbody>
<tr>
<td>OpProtoMake definition </td>
<td>.cc file, Backward Op does not need to define OpProtoMake </td>
</tr>
<tr>
<td>Op definition </td>
<td> .cc file</td>
</tr>
<tr>
<td>Kernel implementation </td>
<td> The CPU and CUDA shared Kernel are implemented in the .h file. Otherwise, the CPU is implemented in the .cc file and the CUDA is implemented in the .cu file. </td>
</tr>
<tr>
<td>Register Op </td>
<td> Op registration is implemented in .cc file; Kernel registered CPU is implemented in .cc file, CUDA is implemented in .cu file</td>
</tr>
</tbody>
</table>


Implementing new ops is added to the directory [paddle/fluid/operators](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/fluid/operators) and the files are named `*_op.h` ( If yes, `*_op.cc`, `*_op.cu` (if any) ends. ** The system automatically builds op and its corresponding Python extension based on the file name. **


Let's take a matrix multiplication operation, that is, [MulOp](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/operators/mul_op.cc) as an example to describe how to write an Operator with Kernel.


## Implementing C++ class


### Defining the ProtoMaker class

The formula for matrix multiplication: $Out = X * Y$, it can be seen that the calculation consists of two inputs and one output.

First define `ProtoMaker` to describe the input and output of the Op, and add a comment:

```cpp
class MulOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  MulOpMaker (OpProto *proto, OpAttrChecker *op_checker)
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

[`MulOpMaker`](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/operators/mul_op.cc#L76-L127) is inherited from `framework::OpProtoAndCheckerMaker`, the constructor contains 2 parameter:

   - `framework::OpProto` : The former stores Op's input and output and parameter properties, which will be used for the generation of the Python API interface.
   - `framework::OpAttrChecker` : The latter is used to check the legality of the parameter properties.

In the constructor, add the input parameter via `AddInput`, add the output parameter via `AddOutput`, and add the comment of Op via `AddComment`. These functions will add the corresponding content to `OpProto`.

The above code adds two inputs `X` and `Y` to `MulOp`, adds an output `Out`, and explains its meaning. Please follow the [naming convention](https://github.com/ PaddlePaddle/Paddle/blob/develop/doc/fluid/dev/name_convention.md) for naming.


Take [`ScaleOp`](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/operators/scale_op.cc#L38-L55) as an example:

```cpp
template <typename AttrType>
class ScaleOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  ScaleOpMaker (OpProto *proto, OpAttrChecker *op_checker)
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

This example has `AddAttr<AttrType>("scale", "...").SetDefault(1.0);` : Increase the `scale` factor as a parameter attribute and set the default value to 1.0.

### Defining the GradProtoMaker class
Each Op must have a corresponding GraProtoMaker. If GradProtoMaker corresponding to the forward Op is not customized, Fluid provides DefaultGradProtoMaker. The default registration will use all input and output, including Input, Output, Output@Grad, etc., using unneeded variables. Will cause waste of memory.
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

### Defining the Operator class

The following defines the definition of MulOp:

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

[`MulOp`](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/operators/mul_op.cc#L22) is inherited from `OperatorWithKernel`. `public` member:

```cpp
Using framework::OperatorWithKernel::OperatorWithKernel;
```

This sentence means that the constructor using the base class 'OperatorWithKernel` can also be written as:

```cpp
MulOp(const std::string &type, const framework::VariableNameMap &inputs,
	  const framework::VariableNameMap &outputs,
	  const framework::AttributeMap &attrs)
  : OperatorWithKernel(type, inputs, outputs, attrs) {}
```

You also need to override the `InferShape` interface. `InferShape` is a const function. You cannot modify the member variable of Op. The parameter is `const framework::InferShapeContext &ctx`. You can get the input and output and attributes through this parameter. Its function is:

  - Do check, report as early as possible: Check if the input data dimension, type, etc. are legal.
  - Set the shape of the output Tensor.

Usually the definitions of the `OpProtoMaker` and `Op` classes are written in the `.cc` file and placed in `.cc` with the registration function described below.

### Defining the OpKernel class

`MulKernel` inherits from `framework::OpKernel` with the following two template parameters:

- `typename DeviceContext`: Indicates the device type. When different devices (CPU, CUDA) share the same Kernel, the template parameters need to be added. If not shared, the template is not added. An example of not sharing is [`OnehotCrossEntropyOpKernel`](https:/ /github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/operators/cross_entropy_op.h#L43).

- `typename T` : Indicates the data type, such as `float`, `double`, etc.

You need to override the `Compute` interface for the `MulKernel` class.

- `Compute` accepts an input parameter: `const framework::ExecutionContext& context`.

- Compared to `InferShapeContext`, `ExecutionContext` adds device types, as well as input and output and property parameters.

- Implement the specific computation logic of `OpKernel` in the `Compute` function.

The input and output of Op can be obtained by `ExecutionContext::Input<T>()` and `ExecutionContext::Output<T>()` respectively.

**Note: ** If the input/output variable type of op is `LoDTensor` (fluid defaults all Tensor defaults to LoDTensor type), please write `ExecutionContext::Input<LoDTensor>()` and `ExecutionContext:: Output<LoDTensor>()`, do not write `ExecutionContext::Input<Tensor>()` and `ExecutionContext::Output<Tensor>()`. Because if the actual variable type is `SelectedRows`, the `Input<Tensor>()` and `Output<Tensor>()` methods will specialize the `SelectedRows` type to `Tensor`, causing a potential error.

Here is the implementation of `MulKernel` and `Compute`:

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

Need to pay attention to: ** Different devices (CPU, CUDA) share an Op definition, otherwise they share the same `OpKernel`, depending on whether the function called by `Compute` supports different devices. **

`MulOp` CPU, CUDA implementation share the same `Kernel`. An example of `OpKernel` not shared can be found at [`OnehotCrossEntropyOpKernel`](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/operators/cross_entropy_op.h#L43).

In order to make the calculation process of `OpKernel` easier, and the CPU and CUDA code can be reused, we usually use the Eigen unsupported Tensor module to implement the `Compute` interface. For how to use the Eigen library in PaddlePaddle, please refer to [Using Documentation](https://github.com/PaddlePaddle/FluidDoc/blob/develop/doc/fluid/dev/use_eigen_cn.md).

At this point, the forward Op implementation is complete. Next, you need to register the op and kernel in the `.cc` file.
The definition of the reverse Op class, the definition of the reverse OpKernel is similar to the forward Op, and will not be described here. **But note that reverse Op does not have `ProtoMaker`**.

### Register Operator

- Register the forward and reverse Op classes in the `.cc` file and register the CPU Kernel.

	```cpp
	namespace ops = paddle::operators;
	REGISTER_OPERATOR(mul, ops::MulOp, ops::MulOpMaker,
				  Paddle::framework::DefaultGradOpDescMaker<true>)
	REGISTER_OPERATOR(mul_grad, ops::MulGradOp)
	REGISTER_OP_CPU_KERNEL(mul, ops::MulKernel<paddle::platform::CPUDeviceContext, float>);
	REGISTER_OP_CPU_KERNEL(mul_grad,
				  ops::MulGradKernel<paddle::platform::CPUDeviceContext, float>);
	```

	In the above code:

		   - `REGISTER_OPERATOR` : Register the `ops::MulOp` class with the type name `mul`, the `ProtoMaker` of this class is `ops::MulOpMaker`, register `ops::MulOpGrad`, and the type name is `mul_grad`.

		   - `REGISTER_OP_CPU_KERNEL` : Register the `ops::MulKernel` class, and specialize the template parameters to `paddle::platform::CPUPlace` and `float`, and similarly, register the `ops::MulGradKernel` class.


- Register the CUDA Kernel in the `.cu` file.
	- Please note that if the implementation of CUDA Kernel is based on the Eigen unsupported module, add the macro definition `#define EIGEN_USE_GPU` at the beginning of `.cu`. The code example is as follows:


	```cpp
	// if use Eigen unsupported module before include head files
	#define EIGEN_USE_GPU

	namespace ops = paddle::operators;
	REGISTER_OP_CUDA_KERNEL(mul, ops::MulKernel<paddle::platform::CUDADeviceContext, float>);
	REGISTER_OP_CUDA_KERNEL(mul_grad,ops::MulGradKernel<paddle::platform::CUDADeviceContext, float>);
	```

### Compile

Run the following command to compile:

```
make mul_op
```

## Binding Python

The system automatically binds Python to the newly added op and links to the generated lib library.

## Implement unit testing

Single measurement includes comparison of forward Op different devices (CPU, CUDA) implementation, comparison of reverse OP different devices (CPU, CUDA) implementation, reverse Op gradient test. The following describes the [`MulOp` unit test](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/tests/unittests/test_mul_op.py).

### Forward Instrument Single Test

The Op unit test is inherited from `OpTest`. More specific unit tests are done in `TestMulOp`. To test the Operator, you need to:

1. Define input, output, and related property parameters in the `setUp` function.
2. Generate random input data.
3. Implement the same calculation logic as the forward operator in the Python script to get the output value, which is compared to the output of the operator forward calculation.
4. The reverse calculation has been automatically integrated into the test framework and the corresponding interface can be called directly.


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

		The above code first imports the dependent package. The following is a detailed explanation of the important variables in the `setUp` function:

		- `self.op_type = "mul" ` : Defines the type, which is the same as the type registered when the operator is registered.
		- `self.inputs` : Define the input, type `numpy.array`, and initialize it.
		- `self.outputs` : Define the output and complete the same calculation logic as the operator in the Python script, returning the results of the Python side.

### Reverse operator single test

In the reverse test:

- `check_grad` is called in `test_check_grad_normal` to use numerical methods to detect gradient correctness and stability.
  - The first parameter `["X", "Y"]` : specifies gradient detection for the input variables `X`, `Y`.
  - The second parameter `"Out"` : specifies the final output target variable `Out` of the forward network.
  - The third parameter `max_relative_error`: specifies the maximum error value that can be tolerated when detecting gradients.

- The `test_check_grad_ingore_x` and `test_check_grad_ingore_y` branches are used to test cases where only one input gradient needs to be calculated.


### Compiling and executing

The new `test_*.py` unit tests in the `python/paddle/fluid/tests/unittests/` directory will be automatically added to the project for compilation.

Please note that ** is different from Op's compile test. It is necessary to compile the entire project** when running unit test, and you need to open `WITH_TESTING`, ie `cmake paddle_dir -DWITH_TESTING=ON`. After the compilation is successful, execute the following command to run the unit test:

```bash
make test ARGS="-R test_mul_op -V"
```

or:

```bash
ctest -R test_mul_op
```

## Precautions

- The name of the type when registering Op needs to be the same as the name of the Op. That is, it is not allowed to register `REGISTER_OPERATOR(B, ...)` in `A_op.cc`, which will cause unit test error.
- If Op does not implement CUDA Kernel, please do not create an empty `*_op.cu`, which will cause unit test errors.
- If multiple Ops depend on some shared functions, you can create files that are not in the `*_op.*` format, such as the `gather.h` file.

### PADDLE_ENFORCE Usage Note

To check the legality of data when implementing Op, you need to use macro definitions such as PADDLE_ENFORCE and PADDLE_ENFORCE_EQ. The basic format is as follows:

```
PADDLE_ENFORCE (expression, error message)
PADDLE_ENFORCE_EQ (comparison object A, comparison object B, error message)
```

If the expression is true, or the comparison object A=B, the check passes, otherwise the program will be terminated and the corresponding error message will be fed back to the user.
In order to ensure that the prompts are friendly and easy to understand, developers need to pay attention to how to use them.

#### General Principles

Any place where PADDLE_ENFORCE and PADDLE_ENFORCE_** are checked must have a detailed explanation of the comments! **Error message ** can't be empty!

#### Tip Information Writing Standard

1. [required] Where is it wrong? Why is it wrong?

	- For example: `ValueError: Mismatched label shape`

2. [optional] What is the expected input? What is the actual input?

	- For example: `Expected labels dimension=1. Received 4.`

3. [optional] Can you give a change?

	- For example: `Suggested Fix: If your classifier expects one-hot encoding label, check your n_classes argument to the estimatorand/or the shape of your label.Otherwise, check the shape of your label.`

If it is not necessary or concise description, you can clearly express the above points and write according to the situation.

#### FAQ Typical problem

1. No error message or error message is too simple to provide effective prompts to the user!

	Problem example 1: Unwritten message
	```
	PADDLE_ENFORCE(ctx->HasInput("X"), "");
	```
	Problem example 2: The prompt message is too simple
	```
	PADDLE_ENFORCE(i != nullptr, "i must be set"); // What i is it?
	```

2. Using developer-defined variable abbreviations in error messages is not easy to understand!

	Example of the problem:
	```
	PADDLE_ENFORCE(forward_pd != nullptr,
	"Fail to find eltwise_fwd_pd in device context"); //eltwise_fwd_pduser may not understand
	```

3. The OP internally calls the illegal interface: If Op appears inside Output = ShareDataWith(Input)
	Example of the problem:
	```cpp
	auto *out = ctx.Output<framework::LoDTensor>("Out");
	auto *in = ctx.Input<framework::LoDTensor>("X");
	out->ShareDataWith(*in);
	```
	If there is Output = ShareDataWith(Input) inside Op, it is equivalent to a hidden edge in the operator graph, which connects Input and Output. This edge cannot be expressed in graph analysis, causing error based on graph optimization.

4. Performance realization of OP implementation
		It called eigen's broadcast, chop and other operations, the performance will be several times worse than the handwritten cuda kernel. At this point, the implementation of cpu can reuse eigen, and the gpu implementation can implement cuda kernel.


#### OP InferShape check message special instructions

- Check input and output variables, please follow the following format
`Input(variable name) of OP name operator should not be null.`

	The correct example:
	```
	PADDLE_ENFORCE(ctx->HasInput("Input"),
						"Input(Input) of LSTMP operator should not be null.");
	```

- Reverse Op input and output check, to write the name of the reverse Op

	The correct example:
	```
	PADDLE_ENFORCE(ctx->HasInput("X"),
							"Input(X) of LoDResetGrad opreator should not be null.");
	```
