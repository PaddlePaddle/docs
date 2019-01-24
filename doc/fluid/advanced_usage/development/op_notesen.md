# Notes on operator development

## Building logic of fluid's op
### 1.Building logic of fluid's op
All Op in Fluid inherits from `OperatorBase`, and all Ops are stateless. Each Op contains only four member variables: type, inputs, outputs, and attributes.

The core method of Op is Run. The Run method requires two resources: data resources and computing resources. These two resources are obtained respectively by `Scope` and `Place`. Inside of the framework, there is a global `DeviceContextPool`, which is used to record the corresponding relationship between `Place` and `DeviceContext`, that is, each `Place` has one and only one `DeviceContext` corresponding to it, and `DeviceContext` stores the computing resources of the current device. For example, for GPU, these resources include `cudnn_handle`, `cublas_handle`, `stream`, etc, all the calculations inside Op (data copy and CUDA Kernel, etc.) must be done in `DeviceContext`.

The Fluid framework is designed to run on a variety of devices and third-party libraries, and some Op implementations may vary depending on the device or third-party libraries. Therefore, Fluid introduced the OpKernel's approach, that is, an Op can have multiple OpKernels, such Op inherits from `OperatorWithKernel`, the representative of such Op is conv, the OpKerne of conv_op is: `GemmConvKernel`, `CUDNNConvOpKernel`, `ConvMKLDNNOpKernel`, and each OpKernel has two data types, double and float. Representatives that do not need OpKernel like `WhileOp` and so on.

Operator inheritance diagram:
![op_inheritance_relation_diagram](../../pics/op_inheritance_relation_diagram.png)

For further information, please refer to: [multi_devices](https://github.com/PaddlePaddle/FluidDoc/tree/develop/doc/fluid/design/multi_devices),[scope](https://github.com/PaddlePaddle/FluidDoc/Blob/develop/doc/fluid/design/concepts/scope.md),[Developer's_Guide_to_Paddle_Fluid](https://github.com/PaddlePaddle/FluidDoc/blob/develop/doc/fluid/getstarted/Developer's_Guide_to_Paddle_Fluid.md )

### 2.Op's registration logic
The registration items for each Operator include:
	```C++
	OpCreator creator_;
	GradOpMakerFN grad_op_maker_;
	proto::OpProto* proto_{nullptr};
	OpAttrChecker* checker_{nullptr};
	InferVarTypeFN infer_var_type_;
	InferShapeFN infer_shape_;
	```

<table>
<thead>
<tr>
<th>Registrations</th>
<th>Type</th>
<th>Description</th>
<th>call</th>
</tr>
</thead>
<tbody>
<tr>
<td>proto::OpProto </td>
<td>Class </td>
<td>Store the input/output/properties/Op type of Op </td>
<td>Call in compile time </td>
</tr>
<tr>
<td>GradOpMakerFN </td>
<td>Functor </td>
<td> Return a set of OpDescs of the reverse Op corresponding to the current Op, because the reverse of the forward Op may have multiple Ops </td>
<td>Call in compile time </td>
</tr>
<tr>
<td>OpAttrChecker </td>
<td>Class </td>
<td>Check the Op's attr </td>
<td>Call in compile time </td>
</tr>
<tr>
<td>InferVarTypeFN </td>
<td>Functor </td>
<td> Used to infer the type of the output Var, such as LoDTensor, SelectedRows, or other </td>
<td>Call in compile time </td>
</tr>
<tr>
<td>InferShapeFN </td>
<td>Functor </td>
<td> Used to infer the Shape of the Output </td>
<td>Divided into compile time and runtime, compile time is called in Python side; If the Op inherits from OperatorWithKernel, the runtime is called when op.run </td>
</tr>
<tr>
<td>OpCreator </td>
<td>Functor </td>
<td>Create a new OperatorBase for each call </td> 
<td>Call in runtime </td>
</tr>
</tbody>
</table>

Usually you need to call REGISTER_OPERATOR when you make comments on Op, which is:
	```
	REGISTER_OPERATOR(op_type,
					  OperatorBase
					  Op_maker_and_checker_maker,
					  Op_grad_opmaker,
					  Op_infer_var_shape,
					  Op_infer_var_type)
	```

**Note: **

1. For all Op, the first three parameters are required, op_type specifies the name of op, OperatorBase is the object of this Op, op_maker_and_checker_maker is the maker of op and the checker of attr in op.
2. If the Op has a reverse, you must have op_grad_opmaker, because in backward, the reverse Op's Maker will be obtained from the forward Op.
3. The framework provides a default op_grad_opmaker:`DefaultGradOpDescMaker`, which will use the input and output of the forward Op as the input of the reverse Op, and the input of the forward Op's gradient as the output of the reverse Op, and copy the properties of the forward Op. **Note: **DefaultGradOpDescMaker will take all the input and output of the forward Op as the reverse Op input, even if this input is not necessary, this will result in we can't do memory optimization for the unused variables.
4. The framework does not provide a default op_infer_var_shape method. If the Op has no OpKernel, you usually need to add the corresponding op_infer_var_shape method. If the Op is OpKernel, you need to implement the `InferShape` method in `OperatorWithKernel`. You don't need to provide the op_infer_var_shape method. For details, see [while_op.cc](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/operators/controlflow/while_op.cc), [conv_op.cc](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/operators/conv_op.cc).
5. The framework does not provide a default op_infer_var_type method, the user needs to add op_infer_var_shape according to the actual situation. Strictly speaking, each Op should register an InferVarType, and op_infer_var_type infers the type and dtype of the output Var according to the type and dtype of the input Var. **Note: **In the Python-side LayerHelper, the create_variable_for_type_inference operation returns the Variable which is LoDTensor. The C++-side InferVarType can modify the type and dtype of the `Variable`.


For more details, please refer to: [How to write a new Op](../new_op.html)

## Write Op Note
### 1.Op can support input and output types
Fluid's Op input and output are `Variable`. From the design point of view, `Variable` can store any type, Op's input and output `Variable` may be any type, usually the `Variable` stored is ` LoDTensor`, `SlelecteRows`.

**note:**

- `context.Input<Tensor>("Input")` often appears in the code, it does not mean that the `Variable` of "Input" is `Tensor`, but the `Tensor` is obtained from `LoDTensor`'s `Variable` of the "Input". If the `Variable` of "Input" is `SelecetedRows`, an error will be reported.
- If "Input" is `SelectedRows`, `context->GetInputDim("Input")` returns `var->Get<SelectedRows>().GetCompleteDims()` instead of `SelectedRows` in `Tensor`'s Dim.

### 2. Do not rewrite the input data inside Op.
Never make any rewrites of the input data inside Op, as there may be other Ops that need to read this data.

### 3.OpKernel needs to register the data type
Currently all OpKernel are required to register double and float data types.

### 4.Op compatibility issue
The modification of Op needs to consider the compatibility problem. To ensure that the previous model can be loaded and run normally after the modification of Op. <font color="#FF0000">**So it is not allowed to add input or output to the existing Op. It is not allowed to subtract the existing properties of Op and modify the default value**</font>.

### 5.ShareDataWith's call
The function of ShareDataWith is to make the two Tensor share the underlying buffer. When calling this operation, special attention should be paid. In the Op, the ShareDataWith cannot be applied to the output of Op. That is, the Tensor of the Op output must be Malloc.

### 6. Sparse gradient parameter's update method
At present, the sparse gradient will first merge the gradient when updating, that is, accumulate the gradient of the same parameter, and then update the parameters and additional parameters (such as velocity).

### 7. Memory optimization
If the reverse of Op does not require all of the input and output of the forward op as its input, please do not use `DefaultGradOpDescMaker`, which will result in RAM or Memory optimization for unused variables.

### 8. Hybrid device's call
Since the GPU is executed asynchronously, the GPU side may not be actually executed after the CPU call returns, so If you create a temporary variable in Op that you need to use the GPU runtime, when the GPU starts running, the temporary variable may have been released on the CPU side, which may cause GPU calculation errors.

Some of the synchronous and asynchronous operations in the GPU:
```
The following device operations are asynchronous with respect to the host:
	Kernel launches;
	Memory copies within a single device's memory;
	Memory copies from host to device of a memory block of 64 KB or less;
	Memory copies performed by functions that are suffixed with Async;
	Memory set function calls.
```

Note on cudaMemCpy and cudaMemCpyAsync:

- If the data transfer is from the GPU side to the non-page locked CPU side, the data transfer will be synchronous, even if an asynchronous copy operation is called.
- If the data is transferred from the CPU side to the CPU side, the data transfer will be synchronous, even if an asynchronous copy operation is called.

For more information, please refer to: [Asynchronous Concurrent Execution](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#asynchronous-concurrent-execution), [API synchronization behavior](https:// Docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html#api-sync-behavior)

## Op Performance Optimization
### 1. Selection of third-party libraries
In the process of writing Op, the operations provided in high performance (such as cudnn, mkldnn, mklml, eigen, etc.) are preferred, but the benchmark must be done. Some operations in the library may be slower in deep learning tasks. Because the operations provided in high-performance libraries (such as eigen, etc.) are more general in terms of performance, they may not be very good in terms of performance. Usually the amount of data in the deep learning model is small, so in some cases some of the high-performance libraries may be provided. The operation speed is slow. For example, all Op (forward and reverse) of the Elementwise series, the Elementwise operation is called more frequently in the model, especially Elementwise_add, which needs to be added after many operations. In the previous implementation, Elementwise_op directly calls the Eigen library. Since the Elementwise operation needs to broadcast the data in many cases, the experiment finds that the Eigen library is slower to broadcast, and the reason for the slowness is in this PR[#6229](https://github.com/PaddlePaddle/Paddle/pull/6229).

### 2.Op performance optimization
The calculation speed of Op is related to the amount of data input. For some Op, different calculation methods can be selected according to the attribute parameters of Shape and Op of the input data. For example, concat_op, when axis>=1, in the process of splicing multiple tensors, you need to make many copies for each tensor. If it is on the GPU, you need to call cudaMemCopy. Relative to the CPU, the GPU belongs to an external device, so each time the GPU is called, there is a certain overhead, and when the number of copies is required, the overhead is more prominent. At present, the implementation of concat_op will select different calling methods according to the Shape and axis values of the input data. If the input tensor is more, and the axis is not equal to 0, the multiple copy operation is converted into a CUDA Kernel to complete; if input tensor Less, and the axis is equal to 0, use the direct copy. The relevant experimental procedure is described in this PR ([#8669](https://github.com/PaddlePaddle/Paddle/pull/8669)).

Since the call of CUDA Kernel has a certain additional overhead, if the CUDA Kernel is called multiple times in Op, it may affect the execution speed of Op. For example, the previous sequence_expand_op contains many CUDA Kernels. Usually, these CUDA Kernels process a small amount of data, so frequent calls to such Kernels will affect the calculation speed of Op. In this case, it is better to combine these small CUDA Kernels into one. This idea is used in the optimization of the sequence_expand_op procedure (related PR[#9289](https://github.com/PaddlePaddle/Paddle/pull/9289)). The optimized sequence_expand_op is about 1 time faster than the previous implementation, the relevant experimental details are introduced in the PR ([#9289](https://github.com/PaddlePaddle/Paddle/pull/9289)).

Reduce the number of copy and sync operations between the CPU and the GPU. For example, the fetch operation will update the model parameters and get a loss after each iteration, and the copy of the data from the GPU to the CPU without the page lock is synchronous, so frequent fetch multiple parameters will cause the model training speed to change. slow.

## Op numerical stability problem
### 1. Some Ops have numerical stability problems
The main reason for numerical stability is that when the program is run multiple times, the order in which the floating-point data is applied may be different, resulting in different final calculation results. The GPU is accelerated by multi-threaded parallel computing, so it is easy to appear that the order of operations on floating-point numbers is not fixed.

At present, it is found that the convolution operation in cudnn, MaxPooling in cudnn, CudaAtomicXX in CUDA, and aggregation of parameter gradients in Reduce mode of ParallelExecutor are not certain.

For this purpose, some FLAGS is added to the Fluid. For example, FLAGS_cudnn_deterministic is used to force cudnn to use the deterministic algorithm, and FLAGS_cpu_deterministic to force the CPU-side calculation to use the deterministic method.

### 2.WITH_FAST_MATH's on and off
If WITH_FAST_MATH is ON, NVCC will use --use_fast_math when compiling Paddle and Egien. This may cause some operations in CUDA to get faster if they lose some precision, such as log, exp, tanh, etc. Some operations are wrong, such as pow operation, please see [torch/DEPRECEATED-torch7-distro#132](https://github.com/torch/DEPRECEATED-torch7-distro/issues/132) for specific reasons.

## Other
### 1. Error message
The Enforce prompt message cannot be empty and needs to be written, because the error message can analyze the cause of the error more quickly and conveniently.

### 2.Op's mathematical formula
If Op has a mathematical formula, be sure to write the mathematical formula in the code and display it in the Doc of the Python API, because the user may need to understand how Paddle implements Op in comparing the calculation results of different frameworks.

**Note: **The formula preview must be previewed before the merge to the develop branch. See [dynamic_lstmp](http://paddlepaddle.org/documentation/docs/en/1.1/api/layers.html#dynamic-lstmp).

### 3. The order of parameters in the Python-side Op interface
The order of the parameters in the Python API is generally ranked by importance, taking fc as an example:
```
def fc(input,
	   size,
	   num_flatten_dims=1,
	   param_attr=None,
	   bias_attr=None,
	   act=None,
	   is_test=False,
	   name=None)
```

