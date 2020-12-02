.. _cn_api_nn_cn_hard_shrink:

hardshrink
-------------------------------
.. py:function:: paddle.nn.functional.hardshrink(x, threshold=0.5, name=None)

hardshrink激活层。计算公式如下：

.. math::

    hardshrink(x)=
        \left\{
        \begin{aligned}
        &x, & & if \ x > threshold \\
        &x, & & if \ x < -threshold \\
        &0, & & if \ others
        \end{aligned}
        \right.

其中，:math:`x` 为输入的 Tensor

参数
::::::::::
    - x (Tensor) - 输入的 ``Tensor`` ，数据类型为：float32、float64。
    - threshold (float, 可选) - hard_shrink激活计算公式中的threshold值。默认值为0.5。
    - name (str, 可选) - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。

返回
::::::::::
    ``Tensor`` ，数据类型和形状同 ``x`` 一致。

代码示例
::::::::::

.. code-block:: python

    import paddle
    import paddle.nn.functional as F

    x = paddle.to_tensor([-1, 0.3, 2.5])
    out = F.hardshrink(x) # [-1., 0., 2.5]
