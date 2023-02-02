# 基本用法


## 一、 @to_static 概览

动静转换（@to_static）通过解析 Python 代码（抽象语法树，下简称：AST） 实现一行代码即可转为静态图功能，即只需在待转化的函数前添加一个装饰器 ``@paddle.jit.to_static`` 。

如下是一个使用 @to_static 装饰器的 ``Model`` 示例：

```python
import paddle
from paddle.jit import to_static

class SimpleNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.linear = paddle.nn.Linear(10, 3)

    # 方式一：装饰 forward 函数（支持训练）
    @to_static
    def forward(self, x, y):
        out = self.linear(x)
        out = out + y
        return out

net = SimpleNet()
# 方式二：(推荐)仅做预测模型导出时，推荐此种用法
net = paddle.jit.to_static(net)  # 动静转换
```

动转静 @to_static 除了支持预测模型导出，还兼容转为静态图子图训练。仅需要在 ``forward`` 函数上添加此装饰器即可，不需要修改任何其他的代码。

基本执行流程如下：

<img src="https://raw.githubusercontent.com/PaddlePaddle/docs/develop/docs/guides/04_dygraph_to_static/images/to_static_train.png" style="zoom:50%" />


### 1.1 动态图 layer 生成 Program

上述样例中的 ``forward`` 函数包含两行组网代码： ``Linear`` 和 ``add`` 操作。以 ``Linear`` 为例，在 Paddle 的框架底层，每个 Paddle 的组网 API 的实现包括两个分支：

```python

class Linear(...):
    def __init__(self, ...):
        # ...(略)

    def forward(self, input):

        if in_dygraph_mode():  # 动态图分支
            core.ops.matmul(input, self.weight, pre_bias, ...)
            return out
        else:                  # 静态图分支
            self._helper.append_op(type="matmul", inputs=inputs, ...)     # <----- 生成一个 Op
            if self.bias is not None:
                self._helper.append_op(type='elementwise_add', ...)       # <----- 生成一个 Op

            return out
```

动态图 ``layer`` 生成 ``Program`` ，其实是开启 ``paddle.enable_static()`` 时，在静态图下逐行执行用户定义的组网代码，依次添加(对应 ``append_op`` 接口) 到默认的主 Program（即 ``main_program`` ） 中。

### 1.2 动态图 Tensor 转为静态图 Variable

上面提到，所有的组网代码都会在静态图模式下执行，以生成完整的 ``Program`` 。**但静态图 ``append_op`` 有一个前置条件必须满足：**

> **前置条件**：append_op() 时，所有的 inputs，outputs 必须都是静态图的 Variable 类型，不能是动态图的 Tensor 类型。


**原因**：静态图下，操作的都是**描述类单元**：计算相关的 ``OpDesc`` ，数据相关的 ``VarDesc`` 。可以分别简单地理解为 ``Program`` 中的 ``Op`` 和 ``Variable`` 。

因此，在动转静时，我们在需要在**某个统一的入口处**，将动态图 ``Layers`` 中 ``Tensor`` 类型（包含具体数据）的 ``Weight`` 、``Bias`` 等变量转换为**同名的静态图 ``Variable``**。

+ ParamBase &rarr; Parameters
+ VarBase &rarr;   Variable

技术实现上，我们选取了框架层面两个地方作为类型**转换的入口**：

+ ``Paddle.nn.Layer`` 基类的 ``__call__`` 函数
    ```python
    def __call__(self, *inputs, **kwargs):
        # param_guard 会对将 Tensor 类型的 Param 和 buffer 转为静态图 Variable
        with param_guard(self._parameters), param_guard(self._buffers):
            # ... forward_pre_hook 逻辑

            outputs = self.forward(*inputs, **kwargs) # 此处为 forward 函数

            # ... forward_post_hook 逻辑

            return outpus
    ```

+ ``Block.append_op`` 函数中，生成 ``Op`` 之前
    ```python
    def append_op(self, *args, **kwargs):
        if in_dygraph_mode():
            # ... (动态图分支)
        else:
            inputs=kwargs.get("inputs", None)
            outputs=kwargs.get("outputs", None)
            # param_guard 会确保将 Tensor 类型的 inputs 和 outputs 转为静态图 Variable
            with param_guard(inputs), param_guard(outputs):
                op = Operator(
                    block=self,
                    desc=op_desc,
                    type=kwargs.get("type", None),
                    inputs=inputs,
                    outputs=outputs,
                    attrs=kwargs.get("attrs", None))
    ```


以上，是动态图转为静态图的两个核心逻辑，总结如下：

+ 动态图 ``layer`` 调用在动转静时会走底层 ``append_op`` 的分支，以生成 ``Program``
+ 动态图 ``Tensor`` 转为静态图 ``Variable`` ，并确保编译期的 ``InferShape`` 正确执行


## 二、 输入层 InputSpec


静态图下，模型起始的 Placeholder 信息是通过 ``paddle.static.data`` 来指定的，并以此作为编译期的 ``InferShape`` 推导起点。

```python
import paddle
# 开启静态图模式
paddle.enable_static()

# placeholder 信息
x = paddle.static.data(shape=[None, 10], dtype='float32', name='x')
y = paddle.static.data(shape=[None, 3], dtype='float32', name='y')

out = paddle.static.nn.fc(x, 3)
out = paddle.add(out, y)
```


动转静代码示例，通过 ``InputSpec`` 设置 ``Placeholder`` 信息：

```python
import paddle
from paddle.jit import to_static

class SimpleNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.linear = paddle.nn.Linear(10, 3)

    # 方式一：在函数定义处装饰
    @to_static
    def forward(self, x, y):
        out = self.linear(x)
        out = out + y
        return out

net = SimpleNet()

# 方式二：(推荐)仅做预测模型导出时，推荐此种用法
x_spec = InputSpec(shape=[None, 10], name='x')
y_spec = InputSpec(shape=[3], name='y')

net = paddle.jit.to_static(net, input_spec=[x_spec, y_spec])  # 动静转换
```


在导出模型时，需要显式地指定输入 ``Tensor`` 的**签名信息**，优势是：


+ 可以指定某些维度为 ``None`` ， 如 ``batch_size`` ，``seq_len`` 维度
+ 可以指定 Placeholder 的 ``name`` ，方面预测时根据 ``name`` 输入数据

> 注：InputSpec 接口的高阶用法，请参看 [【官方文档】InputSpec 功能介绍](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/04_dygraph_to_static/input_spec_cn.html)


## 三、函数转写

在 NLP、CV 领域中，一个模型常包含层层复杂的子函数调用，动转静中是如何实现**只需装饰最外层的 ``forward`` 函数**，就能递归处理所有的函数？

如下是一个模型样例：

```python
import paddle
from paddle.jit import to_static

class SimpleNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.linear = paddle.nn.Linear(10, 3)

    @to_static
    def forward(self, x, y):
        out = self.my_fc(x)       # <---- self.other_func
        out = add_two(out, y)     # <---- other plain func
        return out

    def my_fc(self, x):
        out = self.linear(x)
        return out

# 此函数可以在任意文件
def add_two(x, y):
    out = x + y
    return out

net = SimpleNet()
# 查看转写的代码内容
paddle.jit.set_code_level(100)

x = paddle.zeros([2,10], 'float32')
y = paddle.zeros([3], 'float32')

out = net(x, y)
```

可以通过 ``paddle.jit.set_code_level(100)`` 在执行时打印代码转写的结果到终端，转写代码如下：

```python
def forward(self, x, y):
    out = paddle.jit.dy2static.convert_call(self.my_fc)(x)
    out = paddle.jit.dy2static.convert_call(add_two)(out, y)
    return out

def my_fc(self, x):
    out = paddle.jit.dy2static.convert_call(self.linear)(x)
    return out

def add_two(x, y):
    out = x + y
    return out
```


如上所示，所有的函数调用都会被转写如下形式：

```python
 out = paddle.jit.dy2static.convert_call( self.my_fc )( x )
  ^                    ^                      ^         ^
  |                    |                      |         |
返回列表           convert_call             原始函数    参数列表
```

即使函数定义分布在不同的文件中， ``convert_call`` 函数也会递归地处理和转写所有嵌套的子函数。

## 四、控制流转写

控制流 ``if/for/while`` 的转写和处理是动转静中比较重要的模块，也是动态图模型和静态图模型实现上差别最大的一部分。

**转写上有两个基本原则：**

+ **并非**所有动态图中的 ``if/for/while`` 都会转写为 ``cond_op/while_op``
+ **只有**控制流的判断条件**依赖了 ``Tensor`` **（如 ``shape`` 或 ``value`` ），才会转写为对应 Op


<img src="https://raw.githubusercontent.com/PaddlePaddle/docs/develop/docs/guides/04_dygraph_to_static/images/convert_cond.png" style="zoom:50%" />



### 4.1 IfElse

首先，无论是否会转写为 ``cond_op`` ，动转静都会首先都会对代码进行处理，**转写为 ``cond`` 接口可以接受的写法**

**示例一：不依赖 Tensor 的控制流**

```python
def not_depend_tensor_if(x, label=None):
    out = x + 1
    if label is not None:              # <----- python bool 类型
        out = paddle.nn.functional.cross_entropy(out, label)
    return out

print(to_static(not_depend_tensor_ifw).code)
# 转写后的代码：
"""
def not_depend_tensor_if(x, label=None):
    out = x + 1

    def true_fn_1(label, out):  # true 分支
        out = paddle.nn.functional.cross_entropy(out, label)
        return out

    def false_fn_1(out):        # false 分支
        return out

    out = paddle.jit.dy2static.convert_ifelse(label is not None, true_fn_1,
        false_fn_1, (label, out), (out,), (out,))

    return out
"""
```


**示例二：依赖 Tensor 的控制流**

```python
def depend_tensor_if(x):
    if paddle.mean(x) > 5.:         # <---- Bool Tensor 类型
        out = x - 1
    else:
        out = x + 1
    return out

print(to_static(depend_tensor_if).code)
# 转写后的代码：
"""
def depend_tensor_if(x):
    out = paddle.jit.dy2static.data_layer_not_check(name='out', shape=[-1],
        dtype='float32')

    def true_fn_0(x):      # true 分支
        out = x - 1
        return out

    def false_fn_0(x):     # false 分支
        out = x + 1
        return out

    out = paddle.jit.dy2static.convert_ifelse(paddle.mean(x) > 5.0,
        true_fn_0, false_fn_0, (x,), (x,), (out,))

    return out
"""
```


规范化代码之后，所有的 ``IfElse`` 均转为了如下形式：

```python
 out = convert_ifelse(paddle.mean(x) > 5.0, true_fn_0, false_fn_0, (x,), (x,), (out,))
  ^          ^                   ^             ^           ^        ^      ^      ^
  |          |                   |             |           |        |      |      |
 输出   convert_ifelse          判断条件       true 分支   false 分支  分支输入 分支输入 输出
```


``convert_ifelse`` 是框架底层的函数，在逐行执行用户代码生成 ``Program`` 时，执行到此处时，会根据**判断条件**的类型（ ``bool`` 还是 ``Bool Tensor`` ），自适应决定是否转为 ``cond_op`` 。

```python
def convert_ifelse(pred, true_fn, false_fn, true_args, false_args, return_vars):

    if isinstance(pred, Variable):  # 触发 cond_op 的转换
        return _run_paddle_cond(pred, true_fn, false_fn, true_args, false_args,
                                return_vars)
    else:                           # 正常的 python if
        return _run_py_ifelse(pred, true_fn, false_fn, true_args, false_args)
```


### 4.2 For/While

``For/While`` 也会先进行代码层面的规范化，在逐行执行用户代码时，才会决定是否转为 ``while_op``。

示例一：不依赖 Tensor 的控制流

```python
def not_depend_tensor_while(x):
    a = 1

    while a < 10:           # <---- a is python scalar
        x = x + 1
        a += 1

    return x

print(to_static(not_depend_tensor_while).code)
"""
def not_depend_tensor_while(x):
    a = 1

    def while_condition_0(a, x):
        return a < 10

    def while_body_0(a, x):
        x = x + 1
        a += 1
        return a, x

    [a, x] = paddle.jit.dy2static.convert_while_loop(while_condition_0,
        while_body_0, [a, x])

    return x
"""
```



示例二：依赖 Tensor 的控制流

```python
def depend_tensor_while(x):
    bs = paddle.shape(x)[0]

    for i in range(bs):       # <---- bas is a Tensor
        x = x + 1

    return x

print(to_static(depend_tensor_while).code)
"""
def depend_tensor_while(x):
    bs = paddle.shape(x)[0]
    i = 0

    def for_loop_condition_0(x, i, bs):
        return i < bs

    def for_loop_body_0(x, i, bs):
        x = x + 1
        i += 1
        return x, i, bs

    [x, i, bs] = paddle.jit.dy2static.convert_while_loop(for_loop_condition_0,
        for_loop_body_0, [x, i, bs])
    return x
"""
```


``convert_while_loop`` 的底层的逻辑同样会根据**判断条件是否为``Variable``**来决定是否转为 ``while_op``

## 五、 Parameters 与 Buffers

**什么是 ``Buffers`` 变量？**

+ **Parameters**：``persistable`` 为 ``True`` ，且每个 batch 都被 Optimizer 更新的变量
+ **Buffers**：``persistable`` 为 ``True`` ，``is_trainable = False`` ，不参与更新，但与预测相关；如 ``BatchNorm`` 层中的均值和方差

在动态图模型代码中，若一个 ``paddle.to_tensor`` 接口生成的 ``Tensor`` 参与了最终预测结果的的计算，则此 ``Tensor`` 需要在转换为静态图预测模型时，也需要作为一个 ``persistable`` 的变量保存到 ``.pdparam`` 文件中。

**举一个例子（错误写法）：**

```python
import paddle
from paddle.jit import to_static

class SimpleNet(paddle.nn.Layer):
    def __init__(self, mask):
        super().__init__()
        self.linear = paddle.nn.Linear(10, 3)

        # mask value，此处不会保存到预测模型文件中
        self.mask = mask # 假设为 [0, 1, 1]

    def forward(self, x, y):
        out = self.linear(x)
        out = out + y
        mask = paddle.to_tensor(self.mask)    # <----- 每次执行都转为一个 Tensor
        out = out * mask
        return out
```


**推荐的写法是：**

```python
class SimpleNet(paddle.nn.Layer):
    def __init__(self, mask):
        super().__init__()
        self.linear = paddle.nn.Linear(10, 3)

        # 此处的 mask 会当做一个 buffer Tensor，保存到 .pdparam 文件
        self.mask = paddle.to_tensor(mask) # 假设为 [0, 1, 1]

    def forward(self, x, y):
        out = self.linear(x)
        out = out + y
        out = out * self.mask                 # <---- 直接使用 self.mask
        return out
```


总结一下 ``buffers`` 的用法：

+  若某个非 ``Tensor`` 数据需要当做 ``Persistable`` 的变量序列化到磁盘，则最好在 ``__init__`` 中调用 ``self.XX= paddle.to_tensor(xx)`` 接口转为 ``buffer`` 变量
