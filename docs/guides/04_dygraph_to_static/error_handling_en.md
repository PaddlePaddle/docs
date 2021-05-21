# Error Handling

This section will introduce the error information when an exception occurs, so as to help you better understand the Dynamic-to-Static error information.
When running the transformed static graph code, the internal procedure can be divided into two steps: the dynamic graph code is transformed into the static graph code, and the static graph code is run. We will introduce the error reporting in these two steps.

## Exceptions in Dynamic-to-Static Transformation

If ProgramTranslator cannot transform a function, it will display a warning message and try to run the function as-is.

In the following code, the function `inner_func` is transformed before calling. When calling `inner_func` in `x = inner_func(data)`, it is not allowed to transform repeatedly, and a warning message will be given:

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

The warning message is as follows:
```bash
2020-01-01 00:00:00,104-WARNING: <function inner_func at 0x125b3a550> doesn't have to be transformed to static function because it has been transformed before, it will be run as-is.
```
## Exceptions in Running Transformed Code

When an exception occurs in the transformed code by ProgramTranslator, the exception is caught and the error message is augmented. It maps the error line of the static graph code to the un-transformed dynamic graph code, and then re-raises the exception.

Among the features of the re-raised exception:

- Some useless call stacks of Dynamic-to-Static are hidden;
- For the abnormal information of the transformed code, the prompt "in transformed code:" is given;
- The error message includes references to the original dynamic graph code before transformation;

For example, if executing the following code, an exception is raised when the static graph is built, that is, at compile time:

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

The above error information can be divided into three points:

1. In the error stack, the call stacks related to the code transformation process are hidden by default and not displayed, so as to avoid confusion.

2. In the error message processed by ProgramTranslator, a prompt "In transformed code:" will be included, which means that the following error information stack is raised when running the converted code:

    ```bash
    AssertionError: In transformed code:

        File "<ipython-input-13-f9c3ea702e3a>", line 7, in func (* user code *)
            x = paddle.reshape(x, shape=[-1, -1])
        File "paddle/fluid/layers/nn.py", line 6193, in reshape
            attrs["shape"] = get_attr_shape(shape)
        File "paddle/fluid/layers/nn.py", line 6169, in get_attr_shape
            "be -1. But received shape[%d] is also -1." % dim_idx)
    ```
    `File "<ipython-input-13-f9c3ea702e3a>", line 7, in func` is the location information of un-transformed code, `x = paddle.reshape(x, shape=[-1, -1])` is the un-transformed code.

3. The new exception contains the message that the exception originally reported, as follows:  
    ```bash
    AssertionError: Only one dimension value of 'shape' in reshape can be -1. But received shape[1] is also -1.
    ```  

> **NOTE:**
>
> If you want to view Paddle native error stack, that is, the error stack that has not been processed by Dynamic-to-Static, you can set the environment variable ``TRANSLATOR_DISABLE_NEW_ERROR=1`` to disable the Dynamic-to-Static error handling module. The default value of this environment variable is 0, which means to enable Dynamic-to-Static error handling module.

If execute the following code, an exception is raised when the static graph is executed at runtime:

```Python
@paddle.jit.to_static
def func(x):
    x = paddle.to_tensor(x)
    two = paddle.full(shape=[1], fill_value=2, dtype="int32")
    x = paddle.reshape(x, shape=[1, two])
    return x

func(np.ones([3]).astype("int32"))
```

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

In the above exception, in addition to hiding part of the error stack and locating the error to the un-transformed dynamic graph code, the C++ error stack is hidden. You can set the environment variable `FLAGS_call_stack_level=2` to show C++ stack information.

> **NOTE:**
>
> If you want to view the hidden part of the error stack, you can set the environment variable ``TRANSLATOR_SIMPLIFY_NEW_ERROR=0``. The default value of this environment variable is 1, which means to hide redundant error stack.
