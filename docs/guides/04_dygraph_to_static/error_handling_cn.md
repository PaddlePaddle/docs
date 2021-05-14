# 报错信息处理

本节内容将介绍使用动态图转静态图（下文简称：动转静）功能发生异常时，[ProgramTranslator](./program_translator_cn.html)的动转静报错模块对报错信息做的处理，以帮助您更好地理解动转静报错信息。使用动转静功能运行动态图代码时，内部可以分为2个步骤：动态图代码转换成静态图代码，运行静态图代码。接下来将分别介绍这2个步骤中的异常报错情况。

## 动转静过程中的异常
在动态图代码转换成静态图代码的过程中，如果 ProgramTranslator 无法转换一个函数时，将会显示警告信息，并尝试直接运行该函数。
如下代码中，函数 `inner_func` 在调用前被转换成静态图代码，当 `x = inner_func(data)` 调用该函数时，不能重复转换，会给出警告信息：

```python
import paddle
import numpy as np

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

ProgramTranslator 打印的警告信息如下：

```bash
2020-01-01 00:00:00,104-WARNING: <function inner_func at 0x125b3a550> doesn't have to be transformed to static function because it has been transformed before, it will be run as-is.
```

## 运行转换后的代码报错

如果在动转静后的静态图代码中发生异常，ProgramTranslator 会捕获该异常，增强异常报错信息，将静态图代码报错行映射到转换前的动态图代码，并重新抛出该异常。
重新抛出的异常具有以下特点：

- 隐藏了部分对用户无用的动转静过程调用栈；
- 转换后的代码的异常信息，给出提示"In transformed code:"；
- 报错信息中包含了转换前的原始动态图代码，并给出提示"(* user code *)"；

例如，运行以下代码，在静态图构建时，即编译期会抛出异常：

```python
import paddle
import numpy as np

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
  File "paddle/fluid/dygraph/dygraph_to_static/program_translator.py", line 352, in __call__
    error_data.raise_new_exception()
  File "paddle/fluid/dygraph/dygraph_to_static/error.py", line 188, in raise_new_exception
    raise new_exception
AssertionError: In transformed code:

    File "<ipython-input-13-f9c3ea702e3a>", line 7, in func (* user code *)
        x = paddle.reshape(x, shape=[-1, -1])
    File "paddle/fluid/layers/nn.py", line 6193, in reshape
        attrs["shape"] = get_attr_shape(shape)
    File "paddle/fluid/layers/nn.py", line 6169, in get_attr_shape
        "be -1. But received shape[%d] is also -1." % dim_idx)
    AssertionError: Only one dimension value of 'shape' in reshape can be -1. But received shape[1] is also -1.
```

上述报错信息可以分为3点：

1. 报错栈中，涉及代码转换过程的信息栈默认会被隐藏，不进行展示，以减少干扰信息。

2. ProgramTranslator 处理后的报错信息中，会包含提示 "In transformed code:"，表示之后的报错信息栈，是在运行转换后的代码时的报错信息：

   ```bash
    AssertionError: In transformed code:

        File "<ipython-input-13-f9c3ea702e3a>", line 7, in func (* user code *)
            x = paddle.reshape(x, shape=[-1, -1])
        File "paddle/fluid/layers/nn.py", line 6193, in reshape
            attrs["shape"] = get_attr_shape(shape)
        File "paddle/fluid/layers/nn.py", line 6169, in get_attr_shape
            "be -1. But received shape[%d] is also -1." % dim_idx)
	```
	其中，`File "<ipython-input-13-f9c3ea702e3a>", line 7, in func` 是转换前的代码位置信息，`x = paddle.reshape(x, shape=[-1, -1])` 是转换前用户的动态图代码。

3. 新的异常中，包含原始报错中的的报错信息，如下：
	```bash
	AssertionError: Only one dimension value of 'shape' in reshape can be -1. But received shape[1] is also -1.
	```

> **注解:**
>
> 如果您想查看 Paddle 原生报错信息栈，即未被动转静模块处理过的报错信息栈，可以设置环境变量 ``TRANSLATOR_DISABLE_NEW_ERROR=1`` 关闭动转静报错模块。该环境变量默认值为0，表示默认开启动转静报错模块。

运行以下代码，在静态图运行期会抛出异常：

```Python
@paddle.jit.to_static
def func(x):
    x = paddle.to_tensor(x)
    two = paddle.full(shape=[1], fill_value=2, dtype="int32")
    x = paddle.reshape(x, shape=[1, two])
    return x

func(np.ones([3]).astype("int32"))
```

运行结果：

```bash
Traceback (most recent call last):
  File "<ipython-input-57-c63d6a351262>", line 10, in <module>()
    func(np.ones([3]).astype("int32"))
  File "paddle/fluid/dygraph/dygraph_to_static/program_translator.py", line 352, in __call__
    error_data.raise_new_exception()
  File "paddle/fluid/dygraph/dygraph_to_static/error.py", line 188, in raise_new_exception  
    raise new_exception
EnforceNotMet: In transformed code:

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

    InvalidArgumentError: The 'shape' in ReshapeOp is invalid. The input tensor X'size must be equal to the capacity of 'shape'. But received X's shape = [3], X's size = 3, 'shape' is [1, 2], the capacity of 'shape' is 2.
      [Hint: Expected capacity == in_size, but received capacity:2 != in_size:3.] (at /home/teamcity/work/ef54dc8a5b211854/paddle/fluid/operators/reshape_op.cc:222)
      [Hint: If you need C++ stacktraces for debugging, please set `FLAGS_call_stack_level=2`.]
      [operator < reshape2 > error]  [operator < run_program > error]
```

上述异常中，除了隐藏部分报错栈、报错定位到转换前的动态图代码外，报错信息中隐藏了C++报错栈，您可设置环境变量 `FLAGS_call_stack_level=2` 来展示 C++ 栈信息。

> **注解:**
>
> 如果您想查看被隐藏的信息栈，可以设置环境变量 ``TRANSLATOR_SIMPLIFY_NEW_ERROR=0``。该环境变量默认值为1，表示隐藏冗余的报错信息栈。
