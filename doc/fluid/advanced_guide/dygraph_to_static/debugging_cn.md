# 调试方法
> debug相关的，介绍log、查看转换的代码、pdb调试这些，给出示例。


本节内容将介绍动态图转静态图（下文简称动转静）推荐的几种调试方法。

注意：请确保转换前的动态图代码能够成功运行，建议关闭动转静功能，直接运行动态图，如下：
```
给一段代码，关闭enable_declarative，执行，报错
import paddle
import paddle.fluid as fluid
import numpy as np

@paddle.jit.to_static
def func(x):
    x = paddle.to_tensor(x)
    x = paddle.reshape(x, shape=[-1, -1])
    return x

paddle.disable_static()
prog_trans = fluid.dygraph.ProgramTranslator()
prog_trans.enable(False)
func(np.ones([3, 2]))
```

## 断点调试
您可以使用断点调试代码。
例如，在如下代码中，使用`pdb.set_trace()`：
```Python
import paddle
import numpy as np
import pdb
paddle.disable_static()

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
```Shell
> /tmp/tmpR809hf.py(6)func()
-> def true_fn_0(x):
(Pdb) n
> /tmp/tmpR809hf.py(6)func()
-> def false_fn_0(x):
...
```

如果您想要在原始的动态图代码中使用调试器，请先调用paddle.jit.ProgramTranslator().enable(False)
```
paddle.jit.ProgramTranslator().enable(False)
func(np.ones([3, 2]))
```
运行结果：
```
> <ipython-input-22-0bd4eab35cd5>(10)func()
-> if x > 3:
```

## 打印转换后的代码
您可以打印转换后的静态图代码，有2种方法：
1. 被转换的函数的`code` 属性
```
import paddle
import numpy as np
paddle.disable_static()

@paddle.jit.to_static
def func(x):
    x = paddle.to_tensor(x)
    if x > 3:
        x = x - 1
    return x

print(func.code)
```
运行结果：
```Shell
def func(x):
    x = fluid.layers.assign(x)
    
    def true_fn_0(x):
        x = x - 1
        return x

    def false_fn_0(x):
        return x
    x = fluid.dygraph.dygraph_to_static.convert_operators.convert_ifelse(x >
        3, true_fn_0, false_fn_0, (x,), (x,), (x,))
    return x
```
2.使用`set_code_level` (函数链接)
通过设置`set_code_level`，可以在log中查看转换后的代码
【稍后将这段代码精简，只留最后两行】
```
import paddle
import numpy as np

paddle.disable_static()

@paddle.jit.to_static
def func(x):
    x = paddle.to_tensor(x)
    if x > 3:
        x = x - 1
    return x

paddle.jit.set_code_level()
func(np.ones([1]))
```

运行结果：
```
2020-09-07 17:59:17,980-INFO: After the level 100 ast transformer: 'All Transformers', the transformed code:
def func(x):
    x = fluid.layers.assign(x)

    def true_fn_0(x):
        x = x - 1
        return x

    def false_fn_0(x):
        return x
    x = fluid.dygraph.dygraph_to_static.convert_operators.convert_ifelse(x >
        3, true_fn_0, false_fn_0, (x,), (x,), (x,))
    return x
```
 `set_code_level` 函数可以设置查看不同的AST Transformer转化后的代码，详情请见【链接】。

## print
`print` 函数可以用来查看变量值，该函数在动转静中会被转化。当仅打印Paddle Tensor时，实际执行时会调用paddle.fluid.layers.Print()【链接】，否则仍然执行`print`
```
@paddle.jit.to_static
def func(x):
    x = paddle.to_tensor(x)
    print(x)
    if len(x) > 3:
        x = x - 1
    else:
        x = paddle.ones(shape=[1])
    return x
func(np.ones([1]))
```

运行结果：
```
Variable: assign_0.tmp_0
  - lod: {}
  - place: CPUPlace
  - shape: [1]
  - layout: NCHW
  - dtype: double
  - data: [1]
```

## log打印
我们提供了`paddle.jit.set_verbosity()` 【给个链接】设置日志详细等级，您可以设置并查看不同等级的日志信息，以查看动转静过程中的信息。
可以在代码运行前调用`paddle.jit.set_verbosity()`
```
paddle.jit.set_verbosity(3)
```
运行结果：【修改时间】
```
2020-09-07 19:39:18,123-Level 1:    Source code: 
@paddle.jit.to_static
def func(x):
    x = paddle.to_tensor(x)
    if len(x) > 3:
        x = x - 1
    else:
        x = paddle.ones(shape=[1])
    return x

2020-09-07 19:39:18,152-Level 1: Convert callable object: convert <built-in function len>.
```





