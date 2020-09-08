# 调试方法

本节内容将介绍动态图转静态图（下文简称动转静）推荐的几种调试方法。

注意：请确保转换前的动态图代码能够成功运行，建议使用[fluid.dygraph.ProgramTranslator().enable(False)](../../api_cn/dygraph_cn/ProgramTranslator_cn.rst)关闭动转静功能，直接运行动态图，如下：
```
import paddle
import paddle.fluid as fluid
import numpy as np
paddle.disable_static()
# 关闭动转静动能
fluid.dygraph.ProgramTranslator().enable(False)

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
例如，在代码中，调用`pdb.set_trace()`：
```Python
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

如果您想在原始的动态图代码中使用调试器，请先调用`paddle.jit.ProgramTranslator().enable(False)`，如下：
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

1. 使用被装饰函数的`code` 属性
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

2. 使用`set_code_level(level)`

    通过设置`set_code_level`，可以在log中查看转换后的代码
	```
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

	```Shell
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
 `set_code_level` 函数可以设置查看不同的AST Transformer转化后的代码，详情请见[set_code_level]()<!--TODO：补充set_code_level文档链接-->。

## 使用 `print`
`print` 函数可以用来查看变量，该函数在动转静中会被转化。当仅打印Paddle Tensor时，实际运行时会被转换为Paddle算子[Print](../../api_cn/layers_cn/Print_cn.htm;)，否则仍然运行`print`。
```
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
```
Variable: assign_0.tmp_0
  - lod: {}
  - place: CPUPlace
  - shape: [1]
  - layout: NCHW
  - dtype: double
  - data: [1]
Here call print function.  
```

## log打印
ProgramTranslator在日志中记录了额外的调试信息，以帮助您了解动转静过程中函数是否被成功转换。
您可以调用`paddle.jit.set_verbosity(level)` 设置日志详细等级，并查看不同等级的日志信息。目前，参数`level`可以取值0-3：
- 0: 无日志；
- 1: 包括了动转静转化流程的信息，如转换前的源码、转换的可调用对象；
- 2: 包括以上信息，还包括更详细函数转化日志
- 3: 包括以上信息，以及更详细的动转静日志


可以在代码运行前调用`paddle.jit.set_verbosity()`
```
paddle.jit.set_verbosity(3)
```
运行结果：
```
2020-XX-XX 00:00:00,123-Level 1:    Source code: 
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
