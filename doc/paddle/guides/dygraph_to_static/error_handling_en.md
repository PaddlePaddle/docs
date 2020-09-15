# Error Handling

This section will introduce the error information when an exception occurs, so as to help you better understand the Dynamic-to-Static error information.
When running the transformed static graph code, the internal procedure can be divided into two steps: the dynamic graph code is transformed into the static graph code, and the static graph code is run. We will introduce the error reporting in these two steps.

## Exceptions in Dynamic-to-Static Transformation

If ProgramTranslator cannot transform a function, it will display a warning message and try to run the function as-is.

In the following code, the function `inner_func` is transformed before calling. When calling `inner_func` in `x = inner_func(data)`, it is not allowed to transform repeatedly, and a warning message will be given:

```python
import paddle
import numpy as np

paddle.disable_static()

@paddle.jit.to_static
def func():
    def inner_func(x):
        x_tensor = paddle.to_tensor(x)
        return x_tensor
    data = np.ones([3]).astype("int32")
    x = inner_func(data)
    return x
func()
```

The warning message is as follows:
```bash
WARNING: <function inner_func at 0x7fa9bcaacf50> doesn't have to be transformed to static function because it has been transformed before, it will be run as-is.
```
## Exceptions in Running Transformed Code

When an exception occurs in the transformed code by ProgramTranslator, the exception is caught and the error message is augmented. It maps the error line of the static graph code to the un-transformed dynamic graph code, and then re-raises the exception.

Among the features of the re-raised exception:

- Some useless call stacks of Dynamic-to-Static are hidden;
- A prompt will be given before the un-transformed code: "In User Code:";
- The error message includes references to the original dynamic graph code before transformation;

For example, if executing the following code, an exception is raised when the static graph is built, that is, at compile time:

```python
import paddle
import numpy as np

paddle.disable_static()

@paddle.jit.to_static
def func(x):
    x = paddle.to_tensor(x)
    x = paddle.reshape(x, shape=[-1, -1])
    return x

func(np.ones([3, 2]))
```

```bash
Traceback (most recent call last):
  <ipython-input-13-f9c3ea702e3a> in <module>()
     func(np.ones([3, 2]))
  File "paddle/fluid/dygraph/dygraph_to_static/program_translator.py", line 332, in __call__
    raise new_exception
AssertionError: In user code:

    File "<ipython-input-13-f9c3ea702e3a>", line 7, in func
        x = fluid.layers.reshape(x, shape=[-1, -1])
    File "paddle/fluid/layers/nn.py", line 6193, in reshape
        attrs["shape"] = get_attr_shape(shape)
    File "paddle/fluid/layers/nn.py", line 6169, in get_attr_shape
        "be -1. But received shape[%d] is also -1." % dim_idx)
    AssertionError: Only one dimension value of 'shape' in reshape can be -1. But received shape[1] is also -1.
```

The above error information can be divided into three points:

1. In the error stack, the call stacks related to the code transformation process are hidden by default and not displayed, so as to avoid confusion.

2. In the error message processed by ProgramTranslator, a prompt "In user code:" will be included, which means that the following error stacks contains the original dynamic graph code, that is, the code written by the user:

    ```bash
    AssertionError: In user code:

        File "<ipython-input-13-f9c3ea702e3a>", line 7, in func
           x = fluid.layers.reshape(x, shape=[-1, -1])
        File "paddle/fluid/layers/nn.py", line 6193, in reshape
            attrs["shape"] = get_attr_shape(shape)
        File "paddle/fluid/layers/nn.py", line 6169, in get_attr_shape
            "be -1. But received shape[%d] is also -1." % dim_idx)
    ```
    `File "<ipython-input-13-f9c3ea702e3a>", line 7, in func` is the location information of un-transformed code, `x = fluid.layers.reshape(x, shape=[-1, -1])` is the un-transformed code.

3. The new exception contains the message that the exception originally reported, as follows:  
    ```bash
    AssertionError: Only one dimension value of 'shape' in reshape can be -1. But received shape[1] is also -1.
    ```  

If execute the following code, an exception is raised when the static graph is executed at runtime:

```Python
@paddle.jit.to_static
def func(x):
    x = paddle.to_tensor(x)
    two = paddle.fill_constant(shape=[1], value=2, dtype="int32")
    x = paddle.reshape(x, shape=[1, two])
    return x

func(np.ones([3]).astype("int32"))
```

```bash
Traceback (most recent call last):
  File "<ipython-input-57-c63d6a351262>", line 10, in <module>()
     func(np.ones([3]).astype("int32"))
  File "paddle/fluid/dygraph/dygraph_to_static/program_translator.py", line 332, in __call__
    raise new_exception

EnforceNotMet: In user code:

    File "<ipython-input-57-c63d6a351262>", line 7, in func
      x = paddle.reshape(x, shape=[1, two])
    File "paddle/tensor/manipulation.py", line 1347, in reshape
      return paddle.fluid.layers.reshape(x=x, shape=shape, name=name)
    File "paddle/fluid/layers/nn.py", line 6209, in reshape
      "XShape": x_shape})
    File "paddle/fluid/layer_helper.py", line 43, in append_op
      return self.main_program.current_block().append_op(*args, **kwargs)
    File "paddle/fluid/framework.py", line 2880, in append_op
      attrs=kwargs.get("attrs", None))
    File "paddle/fluid/framework.py", line 1977, in __init__
      for frame in traceback.extract_stack():

--------------------------------------
C++ Traceback (most recent call last):
--------------------------------------
0   paddle::imperative::Tracer::TraceOp(std::string const&, paddle::imperative::NameVarBaseMap const&, paddle::imperative::NameVarBaseMap const&, paddle::framework::AttributeMap, paddle::platform::Place const&, bool)
1   paddle::imperative::OpBase::Run(paddle::framework::OperatorBase const&, paddle::imperative::NameVarBaseMap const&, paddle::imperative::NameVarBaseMap const&, paddle::framework::AttributeMap const&, paddle::platform::Place const&)
2   paddle::imperative::PreparedOp::Run(paddle::imperative::NameVarBaseMap const&, paddle::imperative::NameVarBaseMap const&, paddle::framework::AttributeMap const&)
3   std::_Function_handler<void (paddle::framework::ExecutionContext const&), paddle::framework::OpKernelRegistrarFunctor<paddle::platform::CPUPlace, false, 0ul, paddle::operators::RunProgramOpKernel<paddle::platform::CPUDeviceContext, float> >::operator()(char const*, char const*, int) const::{lambda(paddle::framework::ExecutionContext const&)#1}>::_M_invoke(std::_Any_data const&, paddle::framework::ExecutionContext const&)
4   paddle::operators::RunProgramOpKernel<paddle::platform::CPUDeviceContext, float>::Compute(paddle::framework::ExecutionContext const&) const
5   paddle::framework::Executor::RunPartialPreparedContext(paddle::framework::ExecutorPrepareContext*, paddle::framework::Scope*, long, long, bool, bool, bool)
6   paddle::framework::OperatorBase::Run(paddle::framework::Scope const&, paddle::platform::Place const&)
7   paddle::framework::OperatorWithKernel::RunImpl(paddle::framework::Scope const&, paddle::platform::Place const&) const
8   paddle::framework::OperatorWithKernel::RunImpl(paddle::framework::Scope const&, paddle::platform::Place const&, paddle::framework::RuntimeContext*) const
9   paddle::operators::ReshapeKernel::operator()(paddle::framework::ExecutionContext const&) const
10  paddle::operators::ReshapeOp::ValidateShape(std::vector<int, std::allocator<int> >, paddle::framework::DDim const&)
11  paddle::platform::EnforceNotMet::EnforceNotMet(std::string const&, char const*, int)
12  paddle::platform::GetCurrentTraceBackString()

----------------------
Error Message Summary:
----------------------
InvalidArgumentError: The 'shape' in ReshapeOp is invalid. The input tensor X'size must be equal to the capacity of 'shape'. But received X's shape = [3], X's size = 3, 'shape' is [1, 2], the capacity of 'shape' is 2.
  [Hint: Expected capacity == in_size, but received capacity:2 != in_size:3.] (at /paddle/paddle/fluid/operators/reshape_op.cc:206)
  [operator < reshape2 > error]  [operator < run_program > error]
```

In the above exception, in addition to hiding part of the error stack and locating the error to the un-transformed dynamic graph code, the error information includes the c++ error stack `C++ Traceback` and `Error Message Summary`, which are the exception from C++ and are displayed in Python exception after processing.
