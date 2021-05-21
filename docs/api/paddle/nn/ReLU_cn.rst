.. _cn_api_nn_ReLU:

ReLU
-------------------------------
.. py:class:: paddle.nn.ReLU(name=None)

ReLU激活层（Rectified Linear Unit）。计算公式如下：

.. math::

    ReLU(x) = max(0, x)

其中，:math:`x` 为输入的 Tensor

参数
::::::::::
    - name (str, 可选) - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。

形状:
::::::::::
    - input: 任意形状的Tensor。
    - output: 和input具有相同形状的Tensor。

代码示例
:::::::::

.. code-block:: python

    import paddle

    x = paddle.to_tensor([-2., 0., 1.])
    m = paddle.nn.ReLU()
    out = m(x) # [0., 0., 1.]
