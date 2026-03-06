.. _cn_api_paddle_Tensor_retain_grad:

retain_grad
-------------------------------

.. py:method:: paddle.Tensor.retain_grad(self)

启用此 Tensor 在反向传播过程中计算梯度。对于叶子张量（leaf tensor）该方法是无操作（no-op），因为叶子张量默认会保留梯度。

此方法是 `retain_grads()` 的别名。

**返回**:
    None

**代码示例**:

.. code-block:: python

    >>> import paddle
    >>> x = paddle.to_tensor([1.0, 2.0, 3.0])
    >>> x.stop_gradient = False
    >>> y = x + x
    >>> y.retain_grad()  # 启用梯度计算
    >>> loss = y.sum()
    >>> loss.backward()
    >>> print(y.grad)
    Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=False,
           [2., 2., 2.])
