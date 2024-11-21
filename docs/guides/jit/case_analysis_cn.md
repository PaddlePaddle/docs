# 案例解析


在[【使用样例】](./basic_usage_cn.html)章节介绍了动转静的用法和机制，下面会结合一些具体的模型代码，解答动转静中比较常见的问题。

## 一、 @to_static 放在哪里？


``@to_static`` 装饰器开启动转静功能的唯一接口，支持两种使用方式：

+ **方式一（推荐用法）**：显式地通过 ``model = to_static(model)`` 调用
    ```python
    from paddle.jit import to_static

    model = SimpleNet()
    model = to_static(model, input_spec=[x_spec, y_spec])
    ```


+  **方式二**：在组网代码的 ``forward`` 函数处装饰
    ```python
    class SimpleNet(paddle.nn.Layer):
        def __init__(self, ...):
            # ....

        @to_static
        def forward(self, x, y):
            # ....
            return out
    ```


如果只是进行预测模型导出，推荐使用方式一，**优势在于**：

+ 使用方式更简洁，不用特意去找模型的 ``forward`` 函数在哪
+ 与其他模块解耦，预测模型的导出逻辑可以单独一个模块

**需要注意的地方：**

+ 默认是将 ``model`` 的 ``forward`` 函数作为入口函数

+ 建议模型搭建时，尽量考虑将预测主逻辑放到 ``forward`` 函数中
    + 将训练独有的逻辑放到 子函数 中，通过 ``if self.training`` 来控制
    + 最大程度抽离 **训练和预测** 的逻辑为 **公共子函数**


## 二、何时指定 InputSpec?

在动转静的原理介绍中，静态图 ``Program`` 的生成需要依赖 ``Placeholder`` 信息，此信息可通过两种方式获得：


+ **方式一（推荐）**：在 ``@to_static`` 接口中指定 ``input_spec`` 参数，显式地提供每个输入 ``Variable`` 的 ``Placeholder`` 信息

    ```python
    model = SimpleNet()

    x_spec = InputSpec(shape=[None, 10], name='x')  # 动态 shape
    y_spec = InputSpec(shape=[3], name='y')

    net = paddle.jit.to_static(net, input_spec=[x_spec, y_spec])
    ```

+ **方式二**：输入具体的 ``Tensor(s)`` 数据，显式地执行一次前向，以此 ``Tensor(s)`` 的 ``shape`` 和 ``dtype`` 作为 ``Placeholder`` 信息
    ```python
    # 假设：模型 forward 定义处已经被 @to_static 装饰了
    model = SimpleNet()

    x = paddle.randn([4, 10], 'float32')
    y = paddle.randn([3], 'float32')

    out = model(x, y)     # 执行一次前向，触发 Program 的转换
    paddle.jit.save(model, './simple_net')
    ```

    + **优点**：直接用输入数据，简单方便
    + **缺点**：无法指定动态 shape，如 batch_size，seq_len 等。



> 注：InputSpec 接口的高阶用法，请参看 [【使用 InputSpec 指定模型输入 Tensor 信息】](./basic_usage_cn.html#inputspec)

## 三、内嵌 Numpy 操作？

动态图模型代码中的 ``numpy`` 相关的操作可以转为静态图么？

> **答：不能**。所有与组网相关的 numpy 操作都必须用 paddle 的 API 重新实现。即不支持 Layer &rarr; numpy operations &rarr; Layer 的组网方式。<br>

**原因**：

+ 静态图 ``Program`` 的计算逻辑描述单元是 ``Op``
+ ``numpy`` 操作无法识别为框架的 ``Op`` ，须用 Paddle API 重新实现才可以

**举个例子：**

```python
def forward(self, x):
    out = self.linear(x)  # [bs, 3]

    # 以下将 tensor 转为了 numpy 进行一系列操作
    x_data = x.numpy().astype('float32')  # [bs, 10]
    weight = np.random.randn([10,3])
    mask = paddle.to_tensor(x_data * weight)  # 此处又转回了 Tensor

    out = out * mask
    return out
```


上述样例需要将 ``forward`` 中的所有的 numpy 操作都转为 Paddle API：

```python
def forward(self, x):
    out = self.linear(x)  # [bs, 3]

    weight = paddle.randn([10,3], 'float32')
    mask = x * weight

    out = out * mask
    return out
```


在之前排查的模型中，存在较多中间转为 numpy 的操作，会无法生成完整的 `Program` ，并导致：

+ 模型动转静时报各种奇怪的错误
+ 可以转换成功，但加载模型预测时，可能会报 **Segment Fault** 等错误

**若遇到类似报错，建议排查下模型代码是否存在此类写法。**

## 四、to_tensor() 的使用

``paddle.to_tensor()`` 接口是动态图模型代码中使用比较频繁的一个接口。 ``to_tensor``  功能强大，可以将一个 ``scalar`` ， ``list`` ，``tuple`` ， ``numpy.ndarray`` 转为 ``paddle.Tensor`` 类型。

此接口是动态图独有的接口，在动转静时，会转换为 ``assign`` 接口：

```python
import paddle
import numpy as np

# 动态图
x = paddle.to_tensor(np.array([2,3,4]))

# 动转静后代码
x = paddle.assign(np.array([2,3,4]))
```


**举个比较常见的例子（错误写法）：**

```python
class SimpleNet(paddle.nn.Layer):
    def __init__(self, mask):
        super().__init__()
        self.linear = paddle.nn.Linear(10, 3)
        self.mask = np.array(mask) # 假设为 [0, 1, 1]

    def forward(self, x, y):
        out = self.linear(x)
        out = out + y

        mask = paddle.to_tensor(self.mask)  # <---- 每次都会调用 assign_op
        out = out * mask

        return out
```

**推荐的写法：**

```python
class SimpleNet(paddle.nn.Layer):
    def __init__(self, mask):
        super().__init__()
        self.linear = paddle.nn.Linear(10, 3)
        self.mask = paddle.to_tensor(mask) # <---- 转为 buffers

    def forward(self, x, y):
        out = self.linear(x)
        out = out + y

        out = out * self.mask              # <--- 省去重复的 assign_op，性能更佳

        return out
```


对于 ``to_tensor`` 的使用，建议是：

+ 推荐尽量放到 ``__init__`` 函数中一次性进行初始化


## 五、 建议都继承 nn.Layer


动态图模型常常包含很多嵌套的子网络，建议各个自定义的子网络 ``sublayer`` **无论是否包含了参数，都继承 ``nn.Layer`` **。

从 **Parameters 和 Buffers**  章节可知，有些 ``paddle.to_tensor`` 接口转来的 ``Tensor`` 也可能参与预测逻辑分支的计算，即模型导出时，也需要作为参数序列化保存到 ``.pdiparams`` 文件中。

> **原因**： 若某个 sublayer 包含了 buffer Variables，但却没有继承 ``nn.Layer`` ，则可能导致保存的 ``.pdiparams`` 文件缺失部分重要参数。

**举个例子：**

```python
class SimpleNet:                               # <---- 默认继承自 object
    def __init__(self, mask):
        super().__init__()
        self.linear = paddle.nn.Linear(10, 3)  # <---- Linear 参数永远都不会被更新
        self.mask = paddle.to_tensor(mask)     # <---- mask 可能未保存到 .pdiparams 文件中

    def forward(self, x, y):
        out = self.linear(x)
        out = out + y
        out = out * self.mask
        return out
```

同时，所有继承 ``nn.Layer`` 的 ``sublayer`` 都建议：

+ 重写 ``forward`` 函数，尽量避免重写 ``__call__`` 函数
> ``__call__`` 函数通常会包含框架层面的一些通用的处理逻辑，比如 ``pre_hook`` 和 ``post_hook`` 。重写此函数可能会覆盖框架层面的逻辑。

+  尽量将 ``forward`` 函数作为 sublayers 的调用入口
> 推荐这样写，但动转静也支持对 sublayers 的其他函数转写处理

## 六、 forward 函数推荐写法

### 6.1 默认参数

模型的 ``forward`` 函数的入参可能除了 ``Tensor`` 类型之外，还有很多其他复杂的类型，如 str、float、bool 等非 Tensor 类型。

```python
class SimpleNet(paddle.nn.Layer):
    def __init__(self, mask):
        super().__init__()
        self.linear = paddle.nn.Linear(10, 3)
        self.mask = paddle.to_tensor(mask)

    def forward(self, x, y， cmd='bn', rate=0.1, flag=False):  # <--- 默认参数
        out = self.linear(x)
        out = out + y
        # .... (略)
        out = out * self.mask
        return out
```

关于所有子函数中的非 ``Tensor`` 类型参数：

+ 建议都提供一个默认的取值
+ 建议默认值最好取预测时的值

> **原因**：``jit.save`` 导出预测模型时，提供了 ``input_spec`` 参数用于指定 Placeholder 信息。目前仅支持指定 Tensor 类型信息，**非 Tensor** 类型信息均使用 **函数定义的默认值。**

### 6.2 train 和 infer 分支

模型的 ``forward`` 等子函数常同时包含 **训练** 和 **预测** 两个分支的代码逻辑。

```python
def forward(self, x):
    if self.training:
        out = paddle.mean(x)
    else:
        out = paddle.sum(x)

    return out

model = SimpleNet()
model.eval()           # <---- 一键切换分支，则只会导出 eval 相关的预测分支

jit.save(mode, model_path)
```


推荐使用 ``self.training`` 或其他非 Tensor 类型的 bool 值进行区分。

此 flag 继承自 ``nn.Layer`` ，因此可通过 ``model.train()`` 和 ``model.eval()`` 来全局切换所有 sublayers 的分支状态。

## 七、非 forward 函数导出

`@to_static` 与 `jit.save` 接口搭配也支持导出非 forward 的其他函数，具体使用方式如下：

```python
class SimpleNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.linear = paddle.nn.Linear(10, 3)

    def forward(self, x, y):
        out = self.linear(x)
        out = out + y
        return out

    def another_func(self, x):
        out = self.linear(x)
        out = out * 2
        return out

net = SimpleNet()
# train(net)  # 模型训练

# step 1: 切换到 eval() 模式 （同上）
net.eval()

# step 2: 定义 InputSpec 信息 （同上）
x_spec = InputSpec(shape=[None, 3], dtype='float32', name='x')

# step 3: @to_static 装饰
static_func = to_static(net.another_func, input_spec=[x_spec])

# step 4: 调用 jit.save 接口
net = paddle.jit.save(static_func, path='another_func')
```

使用上的区别主要在于：

+ **`@to_static` 装饰**：导出其他函数时需要显式地用 `@to_static` 装饰，以告知动静转换模块将其识别、并转为静态图 Program；
+ **`save`接口参数**：调用`jit.save`接口时，需将上述被`@to_static` 装饰后的函数作为**参数**；

执行上述代码样例后，在当前目录下会生成三个文件：
```
another_func.pdiparams        // 存放模型中所有的权重数据
another_func.pdimodel         // 存放模型的网络结构
another_func.pdiparams.info   // 存放额外的其他信息
```


> 关于动转静 @to_static 的用法，以及搭配 `paddle.jit.save` 接口导出预测模型的用法案例，可以参考 [使用样例](./basic_usage_cn.html) 。

## 八、再谈控制流

前面[【控制流转写】](./principle_cn.html#kongzhiliuzhuanxie)提到，不论控制流 ``if/for/while`` 语句是否需要转为静态图中的 ``cond_op/while_op`` ，都会先进行代码规范化，如 ``IfElse`` 语句会规范为如下范式：

```python
def true_fn_0(out):
    # ....
    return out

def false_fn_0(out):
    # ....
    return out

out = convert_ifelse(paddle.mean(x) > 5.0, true_fn_0, false_fn_0, (x,), (x,), (out,))
^          ^                   ^             ^           ^        ^      ^      ^
|          |                   |             |           |        |      |      |
输出   convert_ifelse          判断条件       true 分支   false 分支  分支输入 分支输入 输出
```


### 8.1 list 与 DenseTensorArray

当控制流中，出现了 ``list.append`` 类似语法时，情况会有一点点特殊。

Paddle 框架中的 ``cond_op`` 和 [``while_loop``](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/static/nn/while_loop_cn.html#while-loop) 对输入和返回类型有一个要求：
> 输入或者返回类型必须是：LoDTensor 或者 DenseTensorArray <br><br>
> 即：不支持其他非 LoDTensor 类型

因此控制流中类似：

```python
def forward(self, x)：
    bs = paddle.shape(x)[0]
    outs = []                  # <------ list 类型
    for i in range(bs):        # <------ 依赖 Tensor 的 for
        outs.append(x)         # <------ list.append

    return outs
```

**转写之后的代码：**

```python

def forward(x):
    bs = paddle.shape(x)[0]    # <---- bs 是个静态图 Variable, shape = (1, )
    outs = paddle.tensor.create_array(dtype='float32')    # <--- list 转为 DenseTensorArray
    i = 0

    def for_loop_condition_0(outs, bs, i, x):
        return i < bs

    def for_loop_body_0(outs, bs, i, x):
        paddle.tensor.array_write(x=x, i=paddle.tensor.array_length(outs),
            array=outs)                                   # <---- list.append() 转为 array_write
        i += 1
        return outs, bs, i, x

    [outs, bs, i, x] = paddle.jit.dy2static.convert_while_loop(
        for_loop_condition_0, for_loop_body_0, [outs, bs, i, x])

    return outs
```


关于 控制流中包含 ``list`` 相关操作的几点说明：

+ **并非所有**的 list 都会转为 ``DenseTensorArray``
> 只有在此控制流语句是依赖 ``Tensor`` 时，才会触发 ``list`` &rarr; ``DenseTensorArray`` 的转换

+ 暂不支持依赖 Tensor 的控制流中，使用多层嵌套的 ``list.append`` 操作

    ```python
    def forward(x):
        bs = paddle.shape(x)[0]
        outs = [[]]                # <---- 多层嵌套 list

        for i in range(bs):
            outs[0].append(x)

        return outs
    ```

> 因为框架底层的 ``DenseTensorArray = std::vector< LoDTensor >`` ，不支持两层以上 ``vector`` 嵌套


### 8.2 x.shape 与 paddle.shape(x)

模型中比较常见的控制流转写大多数与 ``batch_size`` 或者 ``x.shape`` 相关。

``x.shape[i]`` 的返回值可能是固定的值，也可能是 ``None`` ，表示动态 shape（如 batch_size）。

如果比较明确 ``x.shape[i]`` 对应的是 **动态 shape**，推荐使用 ``paddle.shape(x)[i]``

如上面的例子：

```python
def forward(self, x)：
    bs = paddle.shape(x)[0]        # <---- x.shape[0] 表示 batch_size，动态 shape
    outs = []
    for i in range(bs):
        outs.append(x)

    return outs
```

> 动态 shape 推荐使用 ``paddle.shape(x)[i]`` ，动转静也对 ``x.shape[i]`` 做了很多兼容处理。前者写法出错率可能更低些。

## 九、jit.save 与默认参数


最后一步是预测模型的导出，Paddle 提供了 ``paddle.jit.save`` 接口，搭配 ``@to_static`` 可以导出预测模型。

使用样例如下：

```python
import paddle
import paddle.nn as nn
from paddle.static import InputSpec

IMAGE_SIZE = 784
CLASS_NUM = 10

class LinearNet(nn.Layer):
    def __init__(self):
        super().__init__()
        self._linear = nn.Linear(IMAGE_SIZE, CLASS_NUM)

    def forward(self, x):
        return self._linear(x)

layer = LinearNet()
layer = paddle.jit.to_static(layer, input_spec=[InputSpec(shape=[None, IMAGE_SIZE], dtype='float32')])

path = "example.model/linear"
paddle.jit.save(layer, path)   # <---- Lazy mode, 此处才会触发 Program 的转换
```

> 更多用法可以参考：[【官网文档】jit.save](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/jit/save_cn.html#save)
