# 报错调试

## 一、动转静报错日志
### 1.1 错误日志怎么看
如下是一个动转静报错实例代码：
```python
import paddle
import numpy as np

@paddle.jit.to_static
def func(x):
    two = paddle.full(shape=[1], fill_value=2, dtype="int32")
    x = paddle.reshape(x, shape=[1, two])
    return x

def train():
    x = paddle.to_tensor(np.ones([3]).astype("int32"))
    func(x)

if __name__ == '__main__':
    train()
```
执行后，报错日志如下图：

<img src="https://raw.githubusercontent.com/PaddlePaddle/docs/develop/docs/guides/04_dygraph_to_static/images/dy2stat_error_log.png" style="zoom:45%" />

报错日志从上到下一共可以分为 4 个部分：

- **原生的 Python 报错栈**：如 1 中的前两行所示，表示`/workspace/Paddle/run_dy2stat_error.py`文件第 145 行调用的函数`train()`导致的后续一系列报错。

- **动转静报错栈起始标志位**：`In transformed code`，表示动转静报错信息栈，指运行转换后的代码时的报错信息。实际场景中，可以直接搜索`In transformed code`关键字，从这一行以下开始看报错日志即可。

- **用户代码报错栈**：隐藏了框架层面的无用的报错信息，突出用户代码报错栈。我们在出错代码下添加了波浪线和 HERE 指示词来提示具体的出错位置，并扩展了出错行代码上下文，帮助你快速定位出错位置。如上图 3 中所示，可以看出最后出错的用户代码为`x = paddle.reshape(x, shape=[1, two])`。

- **框架层面报错信息**：提供了静态图组网报错信息。一般可以直接根据最后三行的信息，定位具体是在生成哪个 OpDesc 时报的错误，一般是与执行此 Op 的 infershape 逻辑报的错误。
如上报错信息表明是 reshape Op 出错，出错原因是 tensor x 的 shape 为[3]，将其 reshape 为[1, 2]是不被允许的。

**NOTE**：在某些场景下，会识别报错类型并给出修改建议，如下图所示。`Revise suggestion`下面是出错的排查建议，你可以根据建议对代码进行排查修改。

<img src="https://raw.githubusercontent.com/PaddlePaddle/docs/develop/docs/guides/04_dygraph_to_static/images/revise_suggestion.png" style="zoom:45%" />

### 1.2 报错信息定制化展示
#### 1.2.1 未经动转静报错模块处理的原生报错信息
若你想查看 Paddle 原生报错信息栈，即未被动转静模块处理过的报错信息栈，可以设置环境变量 `TRANSLATOR_DISABLE_NEW_ERROR=1` 关闭动转静报错模块。该环境变量默认值为 0，表示默认开启动转静报错模块。
在 1.1 小节的代码中添加下面的代码即可以查看原生的报错信息：
```python
import os
os.environ["TRANSLATOR_DISABLE_NEW_ERROR"] = '1'
```
可以得到如下的报错信息：

<img src="https://raw.githubusercontent.com/PaddlePaddle/docs/develop/docs/guides/04_dygraph_to_static/images/original_error_log.png" style="zoom:45%" />

#### 1.2.2 C++报错栈
默认会隐藏 C++报错栈，你可设置 C++端的环境变量 `FLAGS_call_stack_level=2` 来显示 C++ 报错栈信息。如可以在终端输入`export FLAGS_call_stack_level=2`来进行设置，之后可以看到 C++端的报错栈：

<img src="https://raw.githubusercontent.com/PaddlePaddle/docs/develop/docs/guides/04_dygraph_to_static/images/c%2B%2B_error_log.png" style="zoom:45%" />

## 二、调试方法
在调试前**请确保转换前的动态图代码能够成功运行**，下面介绍动转静中推荐的几种调试方法。
### 2.1 pdb 调试
pdb 是 Python 中的一个模块，该模块定义了一个交互式 Python 源代码调试器。它支持在源码行间设置断点和单步执行，列出源代码和变量，运行 Python 代码等。
#### 2.1.1 调试步骤

- step1：在想要进行调试的代码前插入`import pdb; pdb.set_trace()`开启 pdb 调试。
    ```python
    import paddle
    import numpy as np

    @paddle.jit.to_static
    def func(x):
        x = paddle.to_tensor(x)
        import pdb; pdb.set_trace()       # <------ 开启 pdb 调试
        two = paddle.full(shape=[1], fill_value=2, dtype="int32")
        x = paddle.reshape(x, shape=[1, two])
        return x

    func(np.ones([3]).astype("int32"))
    ```

- step2：正常运行.py 文件，在终端会出现下面类似结果，在`(Pdb)`位置后输入相应的 pdb 命令进行调试。
    ```
    > /tmp/tmpm0iw5b5d.py(9)func()
    -> two = paddle.full(shape=[1], fill_value=2, dtype='int32')
    (Pdb)
    ```

- step3：在 pdb 交互模式下输入 l、p 等命令可以查看动转静后静态图相应的代码、变量，进而排查相关的问题。
    ```
    > /tmp/tmpm0iw5b5d.py(9)func()
    -> two = paddle.full(shape=[1], fill_value=2, dtype='int32')
    (Pdb) l
      4     import numpy as np
      5     def func(x):
      6         x = paddle.assign(x)
      7         import pdb
      8         pdb.set_trace()
      9  ->     two = paddle.full(shape=[1], fill_value=2, dtype='int32')
     10         x = paddle.reshape(x, shape=[1, two])
     11         return x
    [EOF]
    (Pdb) p x
    var assign_0.tmp_0 : LOD_TENSOR.shape(3,).dtype(int32).stop_gradient(False)
    (Pdb)
    ```

#### 2.1.2 常用命令

<img src="https://raw.githubusercontent.com/PaddlePaddle/docs/develop/docs/guides/04_dygraph_to_static/images/pdb_cmd.png" style="zoom:45%" />

更多 pdb 使用使用方法可以查看 pdb 的[官方文档](https://docs.python.org/zh-cn/3/library/pdb.html)

### 2.2 打印转换后的静态图代码
你可以打印转换后的静态图代码，有 2 种方法：


#### 2.2.1 set_code_level() 或 TRANSLATOR_CODE_LEVEL
通过调用 `set_code_level()` 或设置环境变量 `TRANSLATOR_CODE_LEVEL`，可以在日志中查看转换后的代码：
```python
import paddle
import numpy as np

@paddle.jit.to_static
def func(x):
    x = paddle.to_tensor(x)
    if x > 3:
        x = x - 1
    return x

paddle.jit.set_code_level() # 也可设置 os.environ["TRANSLATOR_CODE_LEVEL"] = '100'，效果相同
func(np.ones([1]))
```
此外，如果你想将转化后的代码也输出到 `sys.stdout` , 可以设置参数 `also_to_stdout` 为 True，否则将仅输出到 `sys.stderr`。 `set_code_level` 函数可以设置查看不同的 AST Transformer 转化后的代码，详情请见 [set_code_level](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/jit/set_code_level_cn.html)。

#### 2.2.2 被装饰后的函数的 code 属性
如下代码中，装饰器@to_static 会将函数 func 转化为一个类对象 StaticFunction，可以使用 StaticFunction 的 code 属性来获得转化后的代码。
```python
import paddle
import numpy as np

@paddle.jit.to_static
def func(x):
    x = paddle.to_tensor(x)
    if x > 3:
        x = x - 1
    return x

func(np.ones([1]))
print(func.code)
```
运行后可以看到动转静后的静态图代码：
```python
def func(x):
    x = paddle.assign(x)

    def true_fn_0(x):
        x = x - 1
        return x

    def false_fn_0(x):
        return x
    x = paddle.jit.dy2static.convert_ifelse(x > 3, true_fn_0, false_fn_0, (
        x,), (x,), (x,))
    return x
```

### 2.3 使用 print 查看变量
print 函数可以用来查看变量，该函数在动转静中会被转化。当仅打印 Paddle Tensor 时，实际运行时会被转换为 Paddle 算子 Print，否则仍然运行 print。

```python
import paddle
import numpy as np

@paddle.jit.to_static
def func(x):
    x = paddle.to_tensor(x)

    # 打印 x，x 是 Paddle Tensor，实际运行时会运行 Paddle Print(x)
    print(x)
    # 打印注释，非 Paddle Tensor，实际运行时仍运行 print
    print("Here call print function.")

    if len(x) > 3:
        x = x - 1
    else:
        x = paddle.ones(shape=[1])
    return x

func(np.ones([1]))
```
运行后可以看到 x 的值：
```
Variable: assign_0.tmp_0
  - lod: {}
  - place: CUDAPlace(0)
  - shape: [1]
  - layout: NCHW
  - dtype: double
  - data: [1]
```
### 2.4 日志打印
动转静在日志中记录了额外的调试信息，以帮助你了解动转静过程中函数是否被成功转换。 你可以调用 `paddle.jit.set_verbosity(level=0, also_to_stdout=False)` 或设置环境变量 `TRANSLATOR_VERBOSITY=level` 来设置日志详细等级，并查看不同等级的日志信息。目前，`level` 可以取值 0-3：

- 0: 无日志
- 1: 包括了动转静转化流程的信息，如转换前的源码、转换的可调用对象
- 2: 包括以上信息，还包括更详细函数转化日志
- 3: 包括以上信息，以及更详细的动转静日志

> **注意：**
> 日志中包括了源代码等信息，请在共享日志前确保它不包含敏感信息。
打印日志的示例代码：
```python
import paddle
import numpy as np
import os

@paddle.jit.to_static
def func(x):
    x = paddle.to_tensor(x)
    if len(x) > 3:
        x = x - 1
    else:
        x = paddle.ones(shape=[1])
    return x

paddle.jit.set_verbosity(3)
# 或者设置 os.environ["TRANSLATOR_VERBOSITY"] = '3'
func(np.ones([1]))
```

运行结果：
```
Sun Sep 26 08:50:20 Dynamic-to-Static INFO: (Level 1) Source code:
@paddle.jit.to_static
def func(x):
    x = paddle.to_tensor(x)
    if len(x) > 3:
        x = x - 1
    else:
        x = paddle.ones(shape=[1])
    return x

Sun Sep 26 08:50:20 Dynamic-to-Static INFO: (Level 1) Convert callable object: convert <built-in function len>.
```
此外，如果你想将日志也输出到 sys.stdout, 可以设置参数 also_to_stdout 为 True，否则将仅输出到 sys.stderr，详情请见 [set_verbosity](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/jit/set_verbosity_cn.html)。


## 三、快速确定问题原因
经过对报错信息的种类进行汇总整理，可以将动转静的问题大致分为如下几个类别：
### 3.1 (NotFound) Input("X")
**报错信息大致如下：**
```
RuntimeError: (NotFound) Input("Filter") of ConvOp should not be null.
    [Hint: Expected ctx->HasInputs("Filter") == true, but received ctx->HasInputs("Filter"):0 != true:1.]
    [operator < conv2d > error]
```
此类问题的原因一般是：
> 执行到报错所在行的 Paddle API 时，某些输入或者 weight 的类型还是动态图的 Tensor，而非静态图的 Variable 或者 Parameter。

**排查建议：**

- 首先确认代码所在的 sublayer 是否继承了 nn.Layer
- 此行代码所在函数是否绕开了 forward 函数，单独调用的（2.1 版本之前）
- 查看是 Tensor 还是 Variable 类型，可以通过 pdb 交互式调试

### 3.2 Expected input_dims[i] == input_dims[0]
**报错信息大致如下：**
```
[Hint: Expected input_dims[i] == input_dims[0], but received input_dims[i]:-1, -1 != input_dims[0]:16, -1.]
    [operator < xxx_op > error]
```
此类问题的原因一般是：
> 逐个 append_op 生成静态图 Program 时，在执行到某个 Paddle API 时，编译期 infershape 不符合要求。

**排查建议：**

- 代码层面，判断是否是上游使用了 reshape 导致 -1 的污染性传播
> 动态图由于执行时 shape 都是已知的，所以 reshape(x, [-1, 0, 128]) 是没有问题的。但静态图组网时都是编译期的 shape（可能为-1），因此使用 reshape 接口时，尽量减少 -1 的使用。

- 可以结合调试技巧，判断是否是某个 API 的输出 shape 在动静态图下有 diff 行为
> 比如某些 Paddle API 动态图下返回的是 1-D Tensor， 但静态图却是始终和输入保持一致，如 ctx->SetOutputDim("Out", ctx->GetInputDim("X"));

### 3.3 desc->CheckGuards() == true
**报错信息大致如下：**
```
[Hint: Expected desc->CheckGuards() == true, but received desc->CheckGuards():0 != true: 1.]
```
此类问题的原因一般是：
> 执行到报错所在行的 Paddle API 时，某些输入或者 weight 的类型还是动态图的 Tensor，而非静态图的 Variable 或者 Parameter.

如下是当前动、静态图对 slice 语法功能的汇总情况：

<img src="https://raw.githubusercontent.com/PaddlePaddle/docs/develop/docs/guides/04_dygraph_to_static/images/slice.png" style="zoom:45%" />

**排查建议：**

- 模型代码是否存在上述复杂的 Tensor slice 切片操作
- 推荐使用 paddle.slice 接口替换复杂的 Tensor slice 操作

### 3.4 Segment Fault
当动转静出现 段错误 时，报错栈信息也会很少，但导致此类问题的原因一般也比较明确。
此类问题的一般原因是：
> 某个 sublayer 未继承 nn.Layer ，同时在\__init__.py 函数中存在 paddle.to_tensor 接口的调用。导致在生成 Program 或者保存模型参数时，在静态图模式下访问了动态图的 Tensor 数据。

**排查建议：**

- 每个 sublayer 是否继承了 nn.Layer

### 3.5 Container 的使用建议
动态图下，提供了如下几种 container 的容器类：

- ParameterList
    ```python
    class MyLayer(paddle.nn.Layer):
        def __init__(self, num_stacked_param):
            super().__init__()

            w1 = paddle.create_parameter(shape=[2, 2], dtype='float32')
            w2 = paddle.create_parameter(shape=[2], dtype='float32')

            # 此用法下，MyLayer.parameters() 返回为空
            self.params = [w1, w2]                            # <----- 错误用法

            self.params = paddle.nn.ParameterList([w1, w2])   # <----- 正确用法
    ```

- LayerList
    ```python
    class MyLayer(paddle.nn.Layer):
        def __init__(self):
            super().__init__()

            layer1 = paddle.nn.Linear(10, 10)
            layer2 = paddle.nn.Linear(10, 16)

            # 此用法下，MyLayer.parameters() 返回为空
            self.linears = [layer1, layer2]                        # <----- 错误用法

            self.linears = paddle.nn.LayerList([layer1, layer2])   # <----- 正确用法
    ```
