.. _cn_api_paddle_distribution_Transform:

Transform
-------------------------------

.. py:class:: paddle.distribution.Transform()

随机变量变换的基类。

``Transform`` 表示将一个随机变量，经过一个或一些列可微且可逆的映射后，变换为另一个随机变量，
并提供变换前后相应概率密度计算方法。主要应用于对一个分布
:ref:`cn_api_distribution_Distribution` 的随机采样结果进行变换。

假设 :math:`X` 为 ``K`` 元随机变量，概率密度为 :math:`p_X(x)`。映射
:math:`f: x \rightarrow y` 为可微且可逆映射，则 :math:`Y` 的概率密度为

.. math::

    p_Y(y) = p_X(f^{-1}(y)) |det J_{f^{-1}}(y)|


其中 :math:`det` 表示计算行列式，:math:`J_{f^{-1}}(y)` 表示 :math:`f^{-1}` 在
:math:`y` 处的雅可比矩阵。

.. math::

    J(y) = \begin{bmatrix}
    {\frac{\partial x_1}{\partial y_1}} &{\frac{\partial x_1}{\partial y_2}}
    &{\cdots} &{\frac{\partial x_1}{\partial y_K}} \\
    {\frac{\partial x_2}{\partial y_1}}  &{\frac{\partial x_2}
    {\partial y_2}}&{\cdots} &{\frac{\partial x_2}{\partial y_K}} \\
    {\vdots} &{\vdots} &{\ddots} &{\vdots}\\
    {\frac{\partial x_K}{\partial y_1}} &{\frac{\partial x_K}{\partial y_2}}
    &{\cdots} &{\frac{\partial x_K}{\partial y_K}}
    \end{bmatrix}

通过上述描述易知，变换 ``Transform`` 主要包含下述三个操作：

    1.正变换( ``forward`` ):

       表示正向变换 :math:`x \rightarrow f(x)` 。

    2.逆变换( ``inverse`` ):

       表示逆向变换 :math:`y \rightarrow f^{-1}(y)` 。

    3.雅可比行列式绝对值的对数( ``log_det_jacobian`` ):

       又可以细分为正变换雅可比行列式绝对值的对数 ``forward_log_det_jacobian`` 和逆变换雅
       可比行列式绝对值的对数 ``inverse_log_det_jacobian``，两者互为负数关系，只实现一种
       即可。

子类通常通过重写如下方法实现变换功能：

    * ``_forward``
    * ``_inverse``
    * ``_forward_log_det_jacobian``
    * ``_inverse_log_det_jacobian`` (可选)

通常情况下，``_forward_log_det_jacobian`` 与 ``_inverse_log_det_jacobian`` 实现其中
一个即可。仅在某些特定情况下，为了追求更高性能以及数值稳定性，需要实现两者。

如果子类变换改变了输入数据形状，还需要重写：

    * ``_forward_shape``
    * ``_inverse_shape``


方法
:::::::::

forward(x)
'''''''''

计算正变换 :math:`y=f(x)` 的结果。

**参数**

- **x** (Tensor) - 正变换输入参数，通常为 :ref:`cn_api_distribution_Distribution`
  的随机采样结果。

**返回**

- **y** (Tensor) - 正变换的计算结果。


inverse(y)
'''''''''

计算逆变换 :math:`x = f^{-1}(y)`

**参数**

- **y** (Tensor) - 逆变换的输入参数。

**返回**

- **x** (Tensor) - 逆变换的计算结果。

forward_log_det_jacobian(x)
'''''''''

计算正变换雅可比行列式绝对值的对数。

如果变换不是一一映射，则雅可比矩阵不存在，返回 ``NotImplementedError`` 。

**参数**

- **x** (Tensor) - 输入参数。

**返回**

- Tensor - 正变换雅可比行列式绝对值的对数。


inverse_log_det_jacobian(y)
'''''''''

计算逆变换雅可比行列式绝对值的对数。

与 ``forward_log_det_jacobian`` 互为负数。

**参数**

- **y** (Tensor) - 输入参数。

**返回**

- Tensor - 逆变换雅可比行列式绝对值的对数。


forward_shape(shape)
'''''''''

推断正变换输出形状。

**参数**

- **shape** (Sequence[int]) - 正变换输入的形状。

**返回**

- Sequence[int] - 正变换输出的形状。


inverse_shape(shape)
'''''''''

推断逆变换输出形状。

**参数**

- **shape** (Sequence[int]) - 逆变换输入的形状。

**返回**

- Sequence[int] - 逆变换输出的形状。
