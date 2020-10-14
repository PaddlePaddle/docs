.. _cn_api_nn_cn_hardsigmoid:

hardsigmoid
-------------------------------

.. py:function:: paddle.nn.functional.hardsigmoid(x, name=None)

hardsigmoid激活层。sigmoid的分段线性逼近激活函数，速度比sigmoid快，详细解释参见 https://arxiv.org/abs/1603.00391。

.. math::

    hardsigmoid(x)=
        \left\{
        \begin{aligned}
        &0, & & \text{if } x \leq -3 \\
        &1, & & \text{if } x \geq 3 \\
        &x/6 + 1/2, & & \text{otherwise}
        \end{aligned}
        \right.

其中，:math:`x` 为输入的 Tensor

参数:
::::::::::
    - x (Tensor) - 输入的 ``Tensor`` ，数据类型为：float32、float64。
    - name (str, 可选) - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。

返回
::::::::::
    ``Tensor`` ，数据类型和形状同 ``x`` 一致。

代码示例
::::::::::

.. code-block:: python

    import paddle
    import paddle.nn.functional as F

    x = paddle.to_tensor([-4., 5., 1.])
    out = F.hardsigmoid(x) # [0., 1., 0.666667]
