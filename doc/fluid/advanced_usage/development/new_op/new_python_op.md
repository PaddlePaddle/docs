# 如何写新的Python OP

PaddlePaddle Fluid通过 `py_func` 接口支持在Python端编写op。 


## py_func接口概述

`py_func` 具体接口为：

```Python
def py_func(func, x, out, backward_func=None, skip_vars_in_backward_input=None):
    pass
```

其中，

- `x` 是Python Op的输入变量，可以是单个 `Variable` 或者 `List[Variable]` 。
- `out` 是Python Op的输出变量，可以是单个 `Variable` 或者 `List[Variable]` 。
- `func` 是Python Op的前向函数。在运行网络前向时，框架会调用 `out = func(*x)` ，根据前向输入 `x` 和前向函数 `func` 计算前向输出 `out`。
- `backward_func` 是Python Op的反向函数。若 `backward_func` 为 `None` ，则该Python Op没有反向计算逻辑；
  若 `backward_func` 不为 `None`，则框架会在运行网路反向时调用 `backward_func` 计算前向输入 `x` 的梯度。
- `skip_vars_in_backward_input` 为反向函数 `backward_func` 中不需要的输入，可以是单个 `Variable` 或者 `List[Variable]` 。


## 如何使用py_func编写Python Op

以下以tanh为例，介绍如何利用 `py_func` 编写Python Op。

- 第一步：定义前向函数和反向函数

前向函数和反向函数均由Python编写。

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

此处我们利用numpy库完成tanh的前向函数和反向函数编写。

```Python
import numpy as np

def my_tanh(x):
    return np.tanh(x)
    
def my_tanh_grad(x, y, dy):
    return np.array(dy) * (1 - np.square(np.array(y)))
```

注意，前向函数和反向函数的输入均是 `LoDTensor` 类型，输出可以是Numpy Array或 `LoDTensor`。
由于 `LoDTensor` 实现了Python的buffer protocol协议，因此我们既可通过 `numpy.array` 直接将 `LoDTensor` 转换为Numpy Array，也可直接将 `LoDTensor` 作为Numpy函数的输入参数。

tanh的反向函数不需要前向输入x，因此我们可定义一个不需要前向输入x的反向函数，并在后续通过 `skip_vars_in_backward_input` 进行排除 :

```Python
def my_tanh_grad_without_x(y, dy):
    return np.array(dy) * (1 - np.square(np.array(y)))
```

- 第二步：创建前向输出变量

我们需调用 `Program.current_block().create_var` 创建前向输出变量。在创建前向输出变量时，必须指明变量的名称name、数据类型dtype和维度shape。

```Python
import paddle.fluid as fluid

def create_tmp_var(program, name, dtype, shape):
    return program.current_block().create_var(name=name, dtype=dtype, shape=shape)
    
in_var = fluid.layers.data(name='input', dtype='float32', shape=[-1, 28, 28])

# 手动创建前向输出变量
out_var = create_tmp_var(fluid.default_main_program(), name='output', dtype='float32', shape=[-1, 28, 28])
```

- 第三步：调用 `py_func` 组建网络

`py_func` 的调用方式为：

```Python
fluid.layers.py_func(func=my_tanh, x=in_var, out=out_var, backward_func=my_tanh_grad)
```

若我们不希望在反向函数输入参数中出现前向输入，则可使用 `skip_vars_in_backward_input` 进行排查，简化反向函数的参数列表。

```Python
fluid.layers.py_func(func=my_tanh, x=in_var, out=out_var, backward_func=my_tanh_grad_without_x,
    skip_vars_in_backward_input=in_var)
```

至此，使用 `py_func` 编写Python Op的步骤结束。我们可以与使用其他Op一样进行网路训练/预测。


## 注意事项

- `py_func` 的前向函数和反向函数内部不应调用 `fluid.layers.xxx` ，因为前向函数和反向函数是在网络运行时调用的，且输入参数均为C++端的 `LoDTensor` ；
  而 `fluid.layers.xxx` 是在组建网络的阶段调用的，且输入参数为Python端的 `Variable` 。
  
- `skip_vars_in_backward_input` 只能跳过前向输入变量和前向输出变量，不能跳过前向输出的梯度。

- 若某个前向输出变量没有梯度，则 `backward_func` 将接收到 `None` 的输入。若某个前向输入变量没有梯度，则我们应在 `backward_func` 中主动返回
  `None`。
  
