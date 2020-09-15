# 报错信息处理

本节内容将介绍使用动态图转静态图（下文简称：动转静）功能发生异常时，[ProgramTranslator](./program_translator_cn.html)对报错信息做的处理，以帮助您更好地理解动转静报错信息。使用动转静功能运行动态图代码时，内部可以分为2个步骤：动态图代码转换成静态图代码，运行静态图代码。接下来将分别介绍这2个步骤中的异常报错情况。

## 动转静过程中的异常
在动态图代码转换成静态图代码的过程中，如果ProgramTranslator无法转换一个函数时，将会显示警告信息，并尝试直接运行该函数。
如下代码中，函数 `inner_func` 在调用前被转换成静态图代码，当 `x = inner_func(data)` 调用该函数时，不能重复转换，会给出警告信息：

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

ProgramTranslator打印的警告信息如下：

```bash
WARNING: <function inner_func at 0x7fa9bcaacf50> doesn't have to be transformed to static function because it has been transformed before, it will be run as-is.
```

## 运行转换后的代码报错

如果在动转静后的静态图代码中发生异常，ProgramTranslator 会捕获该异常，增强异常报错信息，将静态图代码报错行映射到转换前的动态图代码，并重新抛出该异常。
重新抛出的异常具有以下特点：

- 隐藏了部分对用户无用的动转静过程调用栈；
- 转换前的代码会给出提示："In User Code:"；
- 报错信息中包含了转换前的原始动态图代码；

例如，运行以下代码，在静态图构建时，即编译期会抛出异常：

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

运行结果：
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

上述报错信息可以分为3点：

1. 报错栈中，涉及代码转换过程的信息栈默认会被隐藏，不进行展示，以减少干扰信息。

2. ProgramTranslator处理后的报错信息中，会包含提示"In user code:"，表示之后的报错栈中，包含动转静前的动态图代码，即用户写的代码：
	```bash
	AssertionError: In user code:

        File "<ipython-input-13-f9c3ea702e3a>", line 7, in func
	       x = fluid.layers.reshape(x, shape=[-1, -1])
	    File "paddle/fluid/layers/nn.py", line 6193, in reshape
	        attrs["shape"] = get_attr_shape(shape)
	    File "paddle/fluid/layers/nn.py", line 6169, in get_attr_shape
	        "be -1. But received shape[%d] is also -1." % dim_idx)
	```
	其中，`File "<ipython-input-13-f9c3ea702e3a>", line 7, in func` 是转换前的代码位置信息，`x = fluid.layers.reshape(x, shape=[-1, -1])` 是转换前的代码。

3. 新的异常中，包含原始报错中的的报错信息，如下：
	```bash
	AssertionError: Only one dimension value of 'shape' in reshape can be -1. But received shape[1] is also -1.
	```

运行以下代码，在静态图运行时，即运行期会抛出异常：

```Python
@paddle.jit.to_static
def func(x):
    x = paddle.to_tensor(x)
    two = paddle.fill_constant(shape=[1], value=2, dtype="int32")
    x = paddle.reshape(x, shape=[1, two])
    return x

func(np.ones([3]).astype("int32"))
```

运行结果：

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

上述异常中，除了隐藏部分报错栈、报错定位到转换前的动态图代码外，报错信息中包含了C++报错栈 `C++ Traceback` 和 `Error Message Summary`，这是 Paddle 的 C++ 端异常信息，经处理后在 Python 的异常信息中显示。
