.. _cn_api_autograd_PyLayerContext:

PyLayerContext
-------------------------------

.. py:class:: paddle.autograd.PyLayerContext

``PyLayerContext`` 对象能够辅助 :ref:`cn_api_autograd_PyLayer` 实现某些功能。


代码示例
::::::::::::

.. code-block:: python

    import paddle
    from paddle.autograd import PyLayer

    class cus_tanh(PyLayer):
        @staticmethod
        def forward(ctx, x):
            # ctx is a object of PyLayerContext.
            y = paddle.tanh(x)
            ctx.save_for_backward(y)
            return y

        @staticmethod
        def backward(ctx, dy):
            # ctx is a object of PyLayerContext.
            y, = ctx.saved_tensor()
            grad = dy * (1 - paddle.square(y))
            return grad


方法
::::::::::::
save_for_backward(*tensors)
'''''''''

用于暂存 ``backward`` 需要的  ``Tensor``，在 ``backward`` 中调用 ``saved_tensor`` 获取这些 ``Tensor`` 。

.. note::
  这个 API 只能被调用一次，且只能在 ``forward`` 中调用。

**参数**

 - **tensors** (list of Tensor) - 需要被暂存的 ``Tensor``


**返回**

None

**代码示例**

.. code-block:: python

    import paddle
    from paddle.autograd import PyLayer

    class cus_tanh(PyLayer):
        @staticmethod
        def forward(ctx, x):
            # ctx is a context object that store some objects for backward.
            y = paddle.tanh(x)
            # Pass tensors to backward.
            ctx.save_for_backward(y)
            return y

        @staticmethod
        def backward(ctx, dy):
            # Get the tensors passed by forward.
            y, = ctx.saved_tensor()
            grad = dy * (1 - paddle.square(y))
            return grad


saved_tensor()
'''''''''

获取被 ``save_for_backward`` 暂存的 ``Tensor`` 。


**返回**

如果调用 ``save_for_backward`` 暂存了一些 ``Tensor``，则返回这些 ``Tensor``，否则，返回 None。

**代码示例**

.. code-block:: python

    import paddle
    from paddle.autograd import PyLayer

    class cus_tanh(PyLayer):
        @staticmethod
        def forward(ctx, x):
            # ctx is a context object that store some objects for backward.
            y = paddle.tanh(x)
            # Pass tensors to backward.
            ctx.save_for_backward(y)
            return y

        @staticmethod
        def backward(ctx, dy):
            # Get the tensors passed by forward.
            y, = ctx.saved_tensor()
            grad = dy * (1 - paddle.square(y))
            return grad


mark_not_inplace(self, *tensors)
'''''''''

标记一些输入是不需要 inplace 的。
如果 ``forward`` 的输入输出是同一个 ``Tensor`` ，并且这个 ``Tensor`` 被标记为 not_inplace 的。Paddle 会替用户创建一个新的 Tensor 作为输出。
这样可以防止输入的 ``Tensor`` 的 auto grad 信息被错误的篡改。

.. note::
  这个函数最多只能在 ``forward`` 调用一次,并且所有的参数必须是 ``forward`` 输入的 ``Tensor`` 。

**参数**

 - **tensors** (list of Tensor) - 需要标记 not inplace 的 ``Tensor``

**返回**

None

**代码示例**

.. code-block:: python

    import paddle

    class Exp(paddle.autograd.PyLayer):
        @staticmethod
        def forward(ctx, x):
            ctx.mark_not_inplace(x)
            return x

        @staticmethod
        def backward(ctx, grad_output):
            out = grad_output.exp()
            return out

    x = paddle.randn((1, 1))
    x.stop_gradient = False
    attn_layers = []
    for idx in range(0, 2):
        attn_layers.append(Exp())

    for step in range(0, 2):
        a = x
        for j in range(0,2):
            a = attn_layers[j].apply(x)
        a.backward()


mark_non_differentiable(self, *tensors)
'''''''''

标记一些输出是不需要反向的。
如果 ``forward`` 的输入输出是同一个 ``Tensor`` ，并且这个 ``Tensor`` 被标记为 not_inplace 的。Paddle 会替用户创建一个新的 Tensor 作为输出。
将不需要反向的 ``Tensor`` 标记为 non-differentiable，可以提升反向的性能。但是你在 ``backward`` 函数的输入参数中，仍要为其留有反向梯度的位置。
只是这个反向梯度是 1 个全为 0 的、shape 和 ``forward`` 的输出一样的 ``Tensor`` .

.. note::
  这个函数最多只能在 ``forward`` 调用一次,并且所有的参数必须是 ``forward`` 输出的 ``Tensor`` 。

**参数**

 - **tensors** (list of Tensor) - 需要标记不需要反向的 ``Tensor``


**返回**

None

**代码示例**

.. code-block:: python

    import os
    os.environ['FLAGS_enable_eager_mode'] = '1'
    import paddle
    from paddle.autograd import PyLayer
    import numpy as np

    class Tanh(PyLayer):
        @staticmethod
        def forward(ctx, x):
            a = x + x
            b = x + x + x
            ctx.mark_non_differentiable(a)
            return a, b

        @staticmethod
        def backward(ctx, grad_a, grad_b):
            assert np.equal(grad_a.numpy(), paddle.zeros([1]).numpy())
            assert np.equal(grad_b.numpy(), paddle.ones([1], dtype="float64").numpy())
            return grad_b

    x = paddle.ones([1], dtype="float64")
    x.stop_gradient = False
    a, b = Tanh.apply(x)
    b.sum().backward()

set_materialize_grads(self, value)
'''''''''

设置是否要框架来初始化未初始化的反向梯度。默认是 True。
如果设置为 True，框架会将未初始化的反向梯度数据初始化为 0，然后再调用 ``backward`` 函数。
如果设置为 False，框架会将未初始化的反向梯度以 None 向 ``backward`` 函数传递。

.. note::
  这个函数最多只能在 ``forward`` 中调用。

**参数**

 - **value** (bool) - 是否要框架来初始化未初始化的反向梯度


**返回**

None

**代码示例**

.. code-block:: python

    import os
    os.environ['FLAGS_enable_eager_mode'] = '1'
    import paddle
    from paddle.autograd import PyLayer
    import numpy as np

    class Tanh(PyLayer):
        @staticmethod
        def forward(ctx, x):
            return x+x+x, x+x

        @staticmethod
        def backward(ctx, grad, grad2):
            assert np.equal(grad2.numpy(), paddle.zeros([1]).numpy())
            return grad

    class Tanh2(PyLayer):
        @staticmethod
        def forward(ctx, x):
            ctx.set_materialize_grads(False)
            return x+x+x, x+x

        @staticmethod
        def backward(ctx, grad, grad2):
            assert grad2==None
            return grad

    x = paddle.ones([1], dtype="float64")
    x.stop_gradient = False
    Tanh.apply(x)[0].backward()

    x2 = paddle.ones([1], dtype="float64")
    x2.stop_gradient = False
    Tanh2.apply(x2)[0].backward()
