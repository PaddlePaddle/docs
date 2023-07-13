.. _cn_api_autograd_PyLayer:

PyLayer
-------------------------------

.. py:class:: paddle.autograd.PyLayer

Paddle 通过创建 ``PyLayer`` 子类的方式实现 Python 端自定义算子，这个子类必须遵守以下规则：

1. 子类必须包含静态的 ``forward`` 和 ``backward`` 函数，它们的第一个参数必须是 :ref:`cn_api_autograd_PyLayerContext`，如果 ``backward`` 的某个返回值在 ``forward`` 中对应的 ``Tensor`` 是需要梯度，这个返回值必须为 ``Tensor`` 。

2. ``backward`` 除了第一个参数以外，其他参数都是 ``forward`` 函数的输出 ``Tensor`` 的梯度，因此，``backward`` 输入的 ``Tensor`` 的数量必须等于 ``forward`` 输出 ``Tensor`` 的数量。如果你需在 ``backward`` 中使用 ``forward`` 的输入 ``Tensor``，你可以将这些 ``Tensor`` 输入到 :ref:`cn_api_autograd_PyLayerContext` 的 ``save_for_backward`` 方法，之后在 ``backward`` 中使用这些 ``Tensor`` 。

3. ``backward`` 的输出可以是 ``Tensor`` 或者 ``list/tuple(Tensor)``，这些 ``Tensor`` 是 ``forward`` 输出 ``Tensor`` 的梯度。因此，``backward`` 的输出 ``Tensor`` 的个数等于 ``forward`` 输入 ``Tensor`` 的个数。

构建完自定义算子后，通过 ``apply`` 运行算子。


代码示例
::::::::::::

COPY-FROM: paddle.autograd.PyLayer


方法
::::::::::::
forward(ctx, *args, **kwargs)
'''''''''

``forward`` 函数必须被子类重写，它的第一个参数是 :ref:`cn_api_autograd_PyLayerContext` 的对象，其他输入参数的类型和数量任意。

**参数**

 - **\*args** (tuple) - 自定义算子的输入
 - **\*\*kwargs** (dict) - 自定义算子的输入

**返回**

Tensor 或至少包含一个 Tensor 的 list/tuple

**代码示例**

COPY-FROM: paddle.autograd.PyLayer.forward


backward(ctx, *args, **kwargs)
'''''''''

``backward`` 函数的作用是计算梯度，它必须被子类重写，其第一个参数为 :ref:`cn_api_autograd_PyLayerContext` 的对象，其他输入参数为 ``forward`` 输出 ``Tensor`` 的梯度。它的输出 ``Tensor`` 为 ``forward`` 输入 ``Tensor`` 的梯度。

**参数**

 - **\*args** (tuple) - ``forward`` 输出 ``Tensor`` 的梯度。
 - **\*\*kwargs** (dict) - ``forward`` 输出 ``Tensor`` 的梯度。

**返回**

 ``forward`` 输入 ``Tensor`` 的梯度。

**代码示例**

COPY-FROM: paddle.autograd.PyLayer.backward


apply(cls, *args, **kwargs)
'''''''''

构建完自定义算子后，通过 ``apply`` 运行算子。

**参数**

 - **\*args** (tuple) - 自定义算子的输入
 - **\*\*kwargs** (dict) - 自定义算子的输入

**返回**

Tensor 或至少包含一个 Tensor 的 list/tuple

**代码示例**

COPY-FROM: paddle.autograd.PyLayer
