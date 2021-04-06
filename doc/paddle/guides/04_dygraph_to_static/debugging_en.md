# Debugging Methods

This section will introduce several debugging methods recommended by Dynamic Graph to Static Graph (hereafter called Dynamic-to-Staic).

> **NOTE:**
>
> Please ensure that the dynamic graph code before transformation can run successfully. It is recommended to call [paddle.jit.ProgramTranslator().enable(False)](../../api/dygraph/ProgramTranslator_en.html#enable) to disable Dynamic-to-Static, and run dynamic graph code as follows:


```python
import paddle
import numpy as np

# Disable Dynamic-to-Static
paddle.jit.ProgramTranslator().enable(False)

@paddle.jit.to_static
def func(x):
    x = paddle.to_tensor(x)
    if x > 3:
        x = x - 1
    return x

func(np.ones([3, 2]))
```

## Breakpoint Debugging
When using Dynamic-to-Static, you can use breakpoints to debug.

For example, call `pdb.set_trace()` in your code:
```Python
import pdb

@paddle.jit.to_static
def func(x):
    x = paddle.to_tensor(x)
    pdb.set_trace()
    if x > 3:
        x = x - 1
    return x
```
Executing the following code will land the debugger in the transformed static graph code:
```Python
func(np.ones([3, 2]))
```

```bash
> /tmp/tmpR809hf.py(6)func()
-> def true_fn_0(x):
(Pdb) n
> /tmp/tmpR809hf.py(6)func()
-> def false_fn_0(x):
...
```

Calling [`paddle.jit.ProgramTranslator().enable(False)`](../../api/dygraph/ProgramTranslator_en.html#enable) before executing the code will land the debugger in the original dynamic graph code:
```python
paddle.jit.ProgramTranslator().enable(False)
func(np.ones([3, 2]))
```

```bash
> <ipython-input-22-0bd4eab35cd5>(10)func()
-> if x > 3:
...

```

## Print Transformed Code

There are two ways to print the transformed static graph code:

1. Use the attribute `code` of the decorated function

   In the following code, the decorator `paddle.jit.to_static` transforms `func` into a class object `StaticFunction`. You can use the `code` attribute of `StaticFunction` to get the transformed code.
    ```Python
    @paddle.jit.to_static
    def func(x):
        x = paddle.to_tensor(x)
        if x > 3:
            x = x - 1
        return x

    print(func.code)
    ```
    ```bash

    def func(x):
        x = paddle.nn.functional.assign(x)

        def true_fn_0(x):
            x = x - 1
            return x

        def false_fn_0(x):
            return x
        x = paddle.jit.dy2static.convert_ifelse(x > 3, true_fn_0, false_fn_0, (x,), (x,), (x,))
        return x
    ```
2. Call [`set_code_level(level=100, also_to_stdout=False)`](../../../paddle/api/paddle/fluid/dygraph/jit/set_code_level_en.html) or set environment variable `TRANSLATOR_CODE_LEVEL=level`

    You can view the transformed code in the log by calling `set_code_level` or set environment variable `TRANSLATOR_CODE_LEVEL`.

    ```python
    @paddle.jit.to_static
       def func(x):
       x = paddle.to_tensor(x)
       if x > 3:
           x = x - 1
       return x

    paddle.jit.set_code_level() # the same effect to set os.environ["TRANSLATOR_CODE_LEVEL"] = '100'
    func(np.ones([1]))
    ```

    ```bash
    2020-XX-XX 00:00:00,980 Dynamic-to-Static INFO: After the level 100 ast transformer: 'All Transformers', the transformed code:
    def func(x):
        x = paddle.nn.functional.assign(x)

        def true_fn_0(x):
            x = x - 1
            return x

        def false_fn_0(x):
            return x
        x = paddle.jit.dy2static.convert_ifelse(x > 3, true_fn_0, false_fn_0, (x,), (x,), (x,))
        return x
    ```
    In addition, if you want to output the transformed code to ``sys.stdout``, you can set the argument ``also_to_stdout`` to True, otherwise the transformed code is only output to ``sys.stderr``.
    `set_code_level` can set different levels to view the code transformed by different ast transformers. For details, please refer to [set_code_level](../../../paddle/api/paddle/fluid/dygraph/jit/set_code_level_en.html).

## `print`
You can call `print` to view variables. `print` will be transformed when using Dynamic-to-Static. When only Paddle Tensor is printed, `print` will be transformed and call Paddle operator [Print](../../api/paddle/fluid/layers/control_flow/Print_en.html) in runtime. Otherwise, call python `print`.

```python
@paddle.jit.to_static
def func(x):
    x = paddle.to_tensor(x)
    # x is a Paddle Tensor, so it will run Paddle Print(x) actually.
    print(x)

    # The string is not a Paddle Tensor, so it will run print as-is.
    print("Here call print function.")

    if len(x) > 3:
        x = x - 1
    else:
        x = paddle.ones(shape=[1])
    return x

func(np.ones([1]))
```

```bash
Variable: assign_0.tmp_0
  - lod: {}
  - place: CPUPlace
  - shape: [1]
  - layout: NCHW
  - dtype: double
  - data: [1]
Here call print function.  
```

## Log Printing
ProgramTranslator can log additional debugging information to help you know whether the function was successfully transformed or not.

You can call [`paddle.jit.set_verbosity(level=0, also_to_stdout=False)`](../../../paddle/api/paddle/fluid/dygraph/jit/set_verbosity_en.html) or set environment variable `TRANSLATOR_VERBOSITY=level` to enable logging and view logs of different levels. The argument `level` varies from 0 to 3:
- 0: no logging
- 1: includes the information in Dynamic-to-Static tranformation process, such as the source code not transformed, the callable object to transform and so on
- 2: includes above and more detailed function transformation logs
- 3: includes above and extremely verbose logging

> **WARNING:**
>
> The logs includes information such as source code. Please make sure logs don't contain any sensitive information before sharing them.

You can call `paddle.jit.set_verbosity` to control the verbosity level of logs:
```python
paddle.jit.set_verbosity(3)
```
or use the environment variable `TRANSLATOR_VERBOSITY`：
```python
import os
os.environ["TRANSLATOR_VERBOSITY"] = '3'
```

```bash
2020-XX-XX 00:00:00,123 Dynamic-to-Static INFO: (Level 1) Source code:
@paddle.jit.to_static
def func(x):
    x = paddle.to_tensor(x)
    if len(x) > 3:
        x = x - 1
    else:
        x = paddle.ones(shape=[1])
    return x

2020-XX-XX 00:00:00,152 Dynamic-to-Static INFO: (Level 1) Convert callable object: convert <built-in function len>.
```

In addition, if you want to output the logs to ``sys.stdout``, you can set the argument ``also_to_stdout`` to True, otherwise the logs are only output to ``sys.stderr``. For details, please refer to [set_verbosity](../../../paddle/api/paddle/fluid/dygraph/jit/set_verbosity_en.html).
