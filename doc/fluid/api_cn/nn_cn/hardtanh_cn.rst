.. _cn_api_nn_cn_hardtanh:

hardtanh
-------------------------------
.. py:function:: paddle.nn.functional.hardtanh(x, min=-1.0, max=1.0, name=None):

hardtanh激活层（Hardtanh Activation Operator）。计算公式如下：

.. math::

    hardtanh(x)=
        \left\{
        \begin{aligned}
        &max, & & if \ x > max \\
        &min, & & if \ x < min \\
        &x, & & if \ others
        \end{aligned}
        \right.

其中，:math:`x` 为输入的 Tensor

参数
::::::::::
    - x (Tensor) - 输入的 ``Tensor`` ，数据类型为：float32、float64。
    - min (float, 可选) - hardtanh激活计算公式中的min值。默认值为-1。
    - max (float, 可选) - hardtanh激活计算公式中的max值。默认值为1。
    - name (str, 可选) - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。

返回
::::::::::
    ``Tensor`` ，数据类型和形状同 ``x`` 一致。

代码示例
:::::::::

.. code-block:: python

    import paddle
    import paddle.nn.functional as F
    import numpy as np

    paddle.disable_static()

    x = paddle.to_tensor(np.array([-1.5, 0.3, 2.5]))
    out = F.hardtanh(x) # [-1., 0.3, 1.]
