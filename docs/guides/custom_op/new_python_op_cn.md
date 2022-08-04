# 自定义 Python 算子
## 动态图自定义 Python 算子
Paddle 通过 `PyLayer` 接口和`PyLayerContext`接口支持动态图的 Python 端自定义 OP。


### 相关接口概述

`PyLayer` 接口描述如下：

```Python
class PyLayer:
    @staticmethod
    def forward(ctx, *args, **kwargs):
        pass

    @staticmethod
    def backward(ctx, *args, **kwargs):
        pass

    @classmethod
    def apply(cls, *args, **kwargs):
        pass
```

其中，

- `forward` 是自定义 Op 的前向函数，必须被子类重写，它的第一个参数是 `PyLayerContext` 对象，其他输入参数的类型和数量任意。
- `backward` 是自定义 Op 的反向函数，必须被子类重写，其第一个参数为 `PyLayerContext` 对象，其他输入参数为`forward`输出`Tensor`的梯度。它的输出``Tensor``为``forward``输入`Tensor`的梯度。
- `apply` 是自定义 Op 的执行方法，构建完自定义 Op 后，通过 apply 运行 Op。


`PyLayerContext` 接口描述如下：

```Python
class PyLayerContext:
    def save_for_backward(self, *tensors):
        pass

    def saved_tensor(self):
        pass
```

其中，

- `save_for_backward` 用于暂存`backward`需要的`Tensor`，这个 API 只能被调用一次，且只能在``forward``中调用。
- `saved_tensor` 获取被`save_for_backward`暂存的`Tensor`。


### 如何编写动态图 Python Op

以下以 tanh 为例，介绍如何利用 `PyLayer` 编写 Python Op。


- 第一步：创建`PyLayer`子类并定义前向函数和反向函数

前向函数和反向函数均由 Python 编写，可以方便地使用 Paddle 相关 API 来实现一个自定义的 OP。需要遵守以下规则：
  1. `forward`和`backward`都是静态函数，它们的第一个参数是`PyLayerContext`对象。
  2. `backward` 除了第一个参数以外，其他参数都是`forward`函数的输出`Tensor`的梯度，因此，`backward`输入的`Tensor`的数量必须等于`forward`输出`Tensor`的数量。如果您需在`backward`中使用`forward`中的`Tensor`，您可以利用`save_for_backward`和`saved_tensor`这两个方法传递`Tensor`。
  3. `backward`的输出可以是`Tensor`或者`list/tuple(Tensor)`，这些`Tensor`是`forward`输入`Tensor`的梯度。因此，`backward`的输出`Tensor`的个数等于 forward 输入`Tensor`的个数。如果`backward`的某个返回值（梯度）在`forward`中对应的`Tensor`的`stop_gradient`属性为`False`，这个返回值必须是`Tensor`类型。

```Python
import paddle
from paddle.autograd import PyLayer

# 通过创建`PyLayer`子类的方式实现动态图 Python Op
class cus_tanh(PyLayer):
    @staticmethod
    def forward(ctx, x):
        y = paddle.tanh(x)
        # ctx 为 PyLayerContext 对象，可以把 y 从 forward 传递到 backward。
        ctx.save_for_backward(y)
        return y

    @staticmethod
    # 因为 forward 只有一个输出，因此除了 ctx 外，backward 只有一个输入。
    def backward(ctx, dy):
        # ctx 为 PyLayerContext 对象，saved_tensor 获取在 forward 时暂存的 y。
        y, = ctx.saved_tensor()
        # 调用 Paddle API 自定义反向计算
        grad = dy * (1 - paddle.square(y))
        # forward 只有一个 Tensor 输入，因此，backward 只有一个输出。
        return grad
```
- 第二步：通过`apply`方法组建网络。
`apply`的输入为`forward`中除了第一个参数(`ctx`)以外的输入，`apply`的输出即为`forward`的输出。
```Python
data = paddle.randn([2, 3], dtype="float32")
data.stop_gradient = False
# 通过 apply 运行这个 Python 算子
z = cus_tanh.apply(data)
z.mean().backward()

print(data.grad)
```
### 动态图自定义 Python 算子的注意事项
- 为了从`forward`到`backward`传递信息，您可以在`forward`中给`PyLayerContext`添加临时属性，在`backward`中读取这个属性。如果传递`Tensor`推荐使用`save_for_backward`和`saved_tensor`，如果传递非`Tensor`推荐使用添加临时属性的方式。
```Python
import paddle
from paddle.autograd import PyLayer
import numpy as np

class tanh(PyLayer):
    @staticmethod
    def forward(ctx, x1, func1, func2=paddle.square):
        # 添加临时属性的方式传递 func2
        ctx.func = func2
        y1 = func1(x1)
        # 使用 save_for_backward 传递 y1
        ctx.save_for_backward(y1)
        return y1

    @staticmethod
    def backward(ctx, dy1):
        y1, = ctx.saved_tensor()
        # 获取 func2
        re1 = dy1 * (1 - ctx.func(y1))
        return re1

input1 = paddle.randn([2, 3]).astype("float64")
input2 = input1.detach().clone()
input1.stop_gradient = False
input2.stop_gradient = False
z = tanh.apply(x1=input1, func1=paddle.tanh)
```

- forward 的输入和输出的类型任意，但是至少有一个输入和输出为`Tensor`类型。
```Python
# 错误示例
class cus_tanh(PyLayer):
    @staticmethod
    def forward(ctx, x1, x2):
        y = x1+x2
        # y.shape: 列表类型，非 Tensor,输出至少包含一个 Tensor
        return y.shape

    @staticmethod
    def backward(ctx, dy):
        return dy, dy

data = paddle.randn([2, 3], dtype="float32")
data.stop_gradient = False
# 由于 forward 输出没有 Tensor 引发报错
z, y_shape = cus_tanh.apply(data, data)


# 正确示例
class cus_tanh(PyLayer):
    @staticmethod
    def forward(ctx, x1, x2):
        y = x1+x2
        # y.shape: 列表类型，非 Tensor
        return y, y.shape

    @staticmethod
    def backward(ctx, dy):
        # forward 两个 Tensor 输入，因此，backward 有两个输出。
        return dy, dy

data = paddle.randn([2, 3], dtype="float32")
data.stop_gradient = False
z, y_shape = cus_tanh.apply(data, data)
z.mean().backward()

print(data.grad)
```

- 如果 forward 的某个输入为`Tensor`且`stop_gredient = True`，则在`backward`中与其对应的返回值应为`None`。
```Python
class cus_tanh(PyLayer):
    @staticmethod
    def forward(ctx, x1, x2):
        y = x1+x2
        return y

    @staticmethod
    def backward(ctx, dy):
        # x2.stop_gradient=True，其对应梯度需要返回 None
        return dy, None


data1 = paddle.randn([2, 3], dtype="float32")
data1.stop_gradient = False
data2 = paddle.randn([2, 3], dtype="float32")
z = cus_tanh.apply(data1, data2)
fake_loss = z.mean()
fake_loss.backward()
print(data1.grad)
```

- 如果 forward 的所有输入`Tensor`都是`stop_gredient = True`的，则`backward`不会被执行。
```Python
class cus_tanh(PyLayer):
    @staticmethod
    def forward(ctx, x1, x2):
        y = x1+x2
        return y

    @staticmethod
    def backward(ctx, dy):
        return dy, None


data1 = paddle.randn([2, 3], dtype="float32")
data2 = paddle.randn([2, 3], dtype="float32")
z = cus_tanh.apply(data1, data2)
fake_loss = z.mean()
fake_loss.backward()
# 因为 data1.stop_gradient = True、data2.stop_gradient = True，所以 backward 不会被执行。
print(data1.grad is None)
```


## 静态图自定义 Python 算子
Paddle 通过 `py_func` 接口支持静态图的 Python 端自定义 OP。 py_func 的设计原理在于 Paddle 中的 Tensor 可以与 numpy 数组可以方便的互相转换，从而可以使用 Python 中的 numpy API 来自定义一个 Python OP。


### py_func 接口概述

`py_func` 具体接口为：

```Python
def py_func(func, x, out, backward_func=None, skip_vars_in_backward_input=None):
    pass
```

其中，

- `x` 是 Python Op 的输入变量，可以是单个 `Tensor` | `tuple[Tensor]` | `list[Tensor]` 。多个 Tensor 以 tuple[Tensor]或 list[Tensor]的形式传入。
- `out` 是 Python Op 的输出变量，可以是单个 `Tensor` | `tuple[Tensor]` | `list[Tensor]`，也可以是`Numpy Array `。
- `func` 是 Python Op 的前向函数。在运行网络前向时，框架会调用 `out = func(*x)` ，根据前向输入 `x` 和前向函数 `func` 计算前向输出 `out`。在 ``func`` 建议先主动将 Tensor 转换为 numpy 数组，方便灵活的使用 numpy 相关的操作，如果未转换成 numpy，则可能某些操作无法兼容。
- `backward_func` 是 Python Op 的反向函数。若 `backward_func` 为 `None` ，则该 Python Op 没有反向计算逻辑；
  若 `backward_func` 不为 `None`，则框架会在运行网路反向时调用 `backward_func` 计算前向输入 `x` 的梯度。
- `skip_vars_in_backward_input` 为反向函数 `backward_func` 中不需要的输入，可以是单个 `Tensor` | `tuple[Tensor]` | `list[Tensor]` 。


### 如何使用 py_func 编写 Python Op

以下以 tanh 为例，介绍如何利用 `py_func` 编写 Python Op。

- 第一步：定义前向函数和反向函数

前向函数和反向函数均由 Python 编写，可以方便地使用 Python 与 numpy 中的相关 API 来实现一个自定义的 OP。

若前向函数的输入为 `x_1`, `x_2`, ..., `x_n` ，输出为`y_1`, `y_2`, ..., `y_m`，则前向函数的定义格式为：
```Python
def foward_func(x_1, x_2, ..., x_n):
    ...
    return y_1, y_2, ..., y_m
```

默认情况下，反向函数的输入参数顺序为：所有前向输入变量 + 所有前向输出变量 + 所有前向输出变量的梯度，因此对应的反向函数的定义格式为：
```Python
def backward_func(x_1, x_2, ..., x_n, y_1, y_2, ..., y_m, dy_1, dy_2, ..., dy_m):
    ...
    return dx_1, dx_2, ..., dx_n
```

若反向函数不需要某些前向输入变量或前向输出变量，可设置 `skip_vars_in_backward_input` 进行排除（步骤三中会叙述具体的排除方法）。

注：，x_1, ..., x_n 为输入的多个 Tensor，请以 tuple(Tensor)或 list[Tensor]的形式在 py_func 中传入。建议先主动将 Tensor 通过 numpy.array 转换为数组，否则 Python 与 numpy 中的某些操作可能无法兼容使用在 Tensor 上。

此处我们利用 numpy 的相关 API 完成 tanh 的前向函数和反向函数编写。下面给出多个前向与反向函数定义的示例：

```Python
import numpy as np

# 前向函数 1：模拟 tanh 激活函数
def tanh(x):
    # 可以直接将 Tensor 作为 np.tanh 的输入参数
    return np.tanh(x)

# 前向函数 2：将两个 2-D Tenosr 相加，输入多个 Tensor 以 list[Tensor]或 tuple(Tensor)形式
def element_wise_add(x, y):
    # 必须先手动将 Tensor 转换为 numpy 数组，否则无法支持 numpy 的 shape 操作
    x = np.array(x)
    y = np.array(y)

    if x.shape != y.shape:
        raise AssertionError("the shape of inputs must be the same!")

    result = np.zeros(x.shape, dtype='int32')
    for i in range(len(x)):
        for j in range(len(x[0])):
            result[i][j] = x[i][j] + y[i][j]

    return result

# 前向函数 3：可用于调试正在运行的网络（打印值）
def debug_func(x):
    # 可以直接将 Tensor 作为 print 的输入参数
    print(x)

# 前向函数 1 对应的反向函数，默认的输入顺序为：x、out、out 的梯度
def tanh_grad(x, y, dy):
    # 必须先手动将 Tensor 转换为 numpy 数组，否则"+/-"等操作无法使用
    return np.array(dy) * (1 - np.square(np.array(y)))
```

注意，前向函数和反向函数的输入均是 `Tensor` 类型，输出可以是 Numpy Array 或 `Tensor`。
由于 `Tensor` 实现了 Python 的 buffer protocol 协议，因此即可通过 `numpy.array` 直接将 `Tensor` 转换为 numpy Array 来进行操作，也可直接将 `Tensor` 作为 numpy 函数的输入参数。但建议先主动转换为 numpy Array，则可以任意的使用 python 与 numpy 中的所有操作（例如"numpy array 的+/-/shape"）。

tanh 的反向函数不需要前向输入 x，因此我们可定义一个不需要前向输入 x 的反向函数，并在后续通过 `skip_vars_in_backward_input` 进行排除 :

```Python
def tanh_grad_without_x(y, dy):
    return np.array(dy) * (1 - np.square(np.array(y)))
```

- 第二步：创建前向输出变量

我们需调用 `Program.current_block().create_var` 创建前向输出变量。在创建前向输出变量时，必须指明变量的名称 name、数据类型 dtype 和维度 shape。

```Python
import paddle

paddle.enable_static()

def create_tmp_var(program, name, dtype, shape):
    return program.current_block().create_var(name=name, dtype=dtype, shape=shape)

in_var = paddle.static.data(name='input', dtype='float32', shape=[-1, 28, 28])

# 手动创建前向输出变量
out_var = create_tmp_var(paddle.static.default_main_program(), name='output', dtype='float32', shape=[-1, 28, 28])
```

- 第三步：调用 `py_func` 组建网络

`py_func` 的调用方式为：

```Python
paddle.static.nn.py_func(func=tanh, x=in_var, out=out_var, backward_func=tanh_grad)
```

若我们不希望在反向函数输入参数中出现前向输入，则可使用 `skip_vars_in_backward_input` 进行排查，简化反向函数的参数列表。

```Python
paddle.static.nn.py_func(func=tanh, x=in_var, out=out_var, backward_func=tanh_grad_without_x,
    skip_vars_in_backward_input=in_var)
```

至此，使用 `py_func` 编写 Python Op 的步骤结束。我们可以与使用其他 Op 一样进行网路训练/预测。


### 静态图自定义 Python 算子注意事项

- `py_func` 的前向函数和反向函数内部不应调用 `paddle.xx`组网接口 ，因为前向函数和反向函数是在网络运行时调用的，而 `paddle.xx` 是在组建网络的阶段调用 。

- `skip_vars_in_backward_input` 只能跳过前向输入变量和前向输出变量，不能跳过前向输出的梯度。

- 若某个前向输出变量没有梯度，则 `backward_func` 将接收到 `None` 的输入。若某个前向输入变量没有梯度，则我们应在 `backward_func` 中主动返回
  `None`。
