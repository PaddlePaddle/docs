.. _cn_api_autograd_PyLayerContext:

PyLayerContext
-------------------------------

.. py:class:: paddle.autograd.PyLayerContext

``PyLayerContext`` 对象能够辅助 :ref:`cn_api_autograd_PyLayer` 实现某些功能。


**示例代码**

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


.. py:method:: save_for_backward(self, *tensors)

用于暂存 ``backward`` 需要的  ``Tensor`` ，在 ``backward`` 中调用 ``saved_tensor`` 获取这些 ``Tensor`` 。

.. note::
  这个API只能被调用一次，且只能在 ``forward`` 中调用。

参数
::::::::::
 - **tensors** (list of Tensor) - 需要被暂存的 ``Tensor`` 


返回：None

**示例代码**

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


.. py:method:: saved_tensor(self, *tensors)

获取被 ``save_for_backward`` 暂存的 ``Tensor`` 。


返回：如果调用 ``save_for_backward`` 暂存了一些 ``Tensor`` ，则返回这些 ``Tensor`` ，否则，返回 None。

**示例代码**

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
