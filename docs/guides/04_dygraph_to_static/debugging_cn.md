# 调试方法

本节内容将介绍动态图转静态图（下文简称：动转静）推荐的几种调试方法。

> **注解:**
>
> 请确保转换前的动态图代码能够成功运行，建议使用 [paddle.jit.ProgramTranslator().enable(False)](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/jit/ProgramTranslator_cn.html#enable)关闭动转静功能，直接运行动态图，如下：

```python
import paddle
import numpy as np
# 关闭动转静动能
paddle.jit.ProgramTranslator().enable(False)

@paddle.jit.to_static
def func(x):
    x = paddle.to_tensor(x)
    if x > 3:
        x = x - 1
    return x

func(np.ones([3, 2]))
```

## 断点调试
使用动转静功能时，您可以使用断点调试代码。
例如，在代码中，调用 `pdb.set_trace()`：
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
执行以下代码，将会在转化后的静态图代码中使用调试器：
```Python
func(np.ones([3, 2]))
```

运行结果：
```bash
> /tmp/tmpR809hf.py(6)func()
-> def true_fn_0(x):
(Pdb) n
> /tmp/tmpR809hf.py(6)func()
-> def false_fn_0(x):
...
```

如果您想在原始的动态图代码中使用调试器，请先调用 [`paddle.jit.ProgramTranslator().enable(False)`](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/jit/ProgramTranslator_cn.html#enable)，如下：
```python
paddle.jit.ProgramTranslator().enable(False)
func(np.ones([3, 2]))
```
运行结果：
```bash
> <ipython-input-22-0bd4eab35cd5>(10)func()
-> if x > 3:
...

```

## 打印转换后的代码
您可以打印转换后的静态图代码，有2种方法：

1. 使用被装饰后的函数的 `code` 属性

   如下代码中，装饰器 `paddle.jit.to_static` 会将函数 `func` 转化为一个类对象 `StaticFunction`，可以使用 StaticFunction 的 `code` 属性来获得转化后的代码。
    ```Python
    @paddle.jit.to_static
    def func(x):
        x = paddle.to_tensor(x)
        if x > 3:
            x = x - 1
        return x

    print(func.code)
    ```
    运行结果：

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

2. 使用 [`set_code_level(level=100, also_to_stdout=False)`](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/jit/set_code_level_cn.html) 或环境变量 `TRANSLATOR_CODE_LEVEL=level`

    通过调用 `set_code_level` 或设置环境变量 `TRANSLATOR_CODE_LEVEL`，可以在日志中查看转换后的代码：

    ```python
    @paddle.jit.to_static
       def func(x):
       x = paddle.to_tensor(x)
       if x > 3:
           x = x - 1
       return x

    paddle.jit.set_code_level() # 也可设置 os.environ["TRANSLATOR_CODE_LEVEL"] = '100'，效果相同
    func(np.ones([1]))
    ```
   运行结果：

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
    此外，如果您想将转化后的代码也输出到 ``sys.stdout``, 可以设置参数 ``also_to_stdout`` 为 True，否则将仅输出到 ``sys.stderr``。
    `set_code_level` 函数可以设置查看不同的 AST Transformer 转化后的代码，详情请见 [set_code_level](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/jit/set_code_level_cn.html)。

## 使用 `print`
`print` 函数可以用来查看变量，该函数在动转静中会被转化。当仅打印 Paddle Tensor 时，实际运行时会被转换为 Paddle 算子 [Print](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/static/Print_cn.html)，否则仍然运行 `print`。
```python
@paddle.jit.to_static
def func(x):
    x = paddle.to_tensor(x)

    # 打印x，x是Paddle Tensor，实际运行时会运行Paddle Print(x)
    print(x)

    # 打印注释，非Paddle Tensor，实际运行时仍运行print
    print("Here call print function.")

    if len(x) > 3:
        x = x - 1
    else:
        x = paddle.ones(shape=[1])
    return x

func(np.ones([1]))
```

运行结果：
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

## 日志打印
ProgramTranslator在日志中记录了额外的调试信息，以帮助您了解动转静过程中函数是否被成功转换。
您可以调用 [`paddle.jit.set_verbosity(level=0, also_to_stdout=False)`]((https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/jit/set_verbosity_cn.html)) 或设置环境变量 `TRANSLATOR_VERBOSITY=level` 来设置日志详细等级，并查看不同等级的日志信息。目前，`level` 可以取值0-3：
- 0: 无日志
- 1: 包括了动转静转化流程的信息，如转换前的源码、转换的可调用对象
- 2: 包括以上信息，还包括更详细函数转化日志
- 3: 包括以上信息，以及更详细的动转静日志

> **注意:**
>
> 日志中包括了源代码等信息，请在共享日志前确保它不包含敏感信息。

可以在代码运行前调用 `paddle.jit.set_verbosity` 控制日志详细程度：
```python
paddle.jit.set_verbosity(3)
```
或者设置环境变量 `TRANSLATOR_VERBOSITY`：
```python
import os
os.environ["TRANSLATOR_VERBOSITY"] = '3'
```

运行结果：
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
此外，如果您想将日志也输出到 ``sys.stdout``, 可以设置参数 ``also_to_stdout`` 为 True，否则将仅输出到 ``sys.stderr``，详情请见 [set_verbosity](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/jit/set_verbosity_cn.html)。
