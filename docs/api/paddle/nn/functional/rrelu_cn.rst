.. _cn_api_nn_cn_prelu:

rrelu
-------------------------------

.. py:function:: paddle.nn.functional.rrelu(x, lower=1. / 8., upper=1. / 3., training=True, name=None)

rrelu激活层（RRelu Activation Operator）。计算公式如下：

.. math::

        \text{RReLU}(x) =
                \begin{cases}
                x & \text{if } x \geq 0 \\
                ax & \text{ otherwise }
                \end{cases}

其中，:math:`x` 为输入的 Tensor

参数
::::::::::
    - lower (float, 可选) - 负值斜率的随机值范围下限，`lower`包含在范围中。支持的数据类型：float。默认值为0.125。
    - upper (float, 可选) - 负值斜率的随机值范围上限，`upper`包含在范围中。支持的数据类型：float。默认值为0.333。
    - training (bool): 标记是否为训练阶段。 默认: True。
    - name (str, 可选) - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。

返回
::::::::::
    ``Tensor`` ，数据类型和形状同 ``x`` 一致。

代码示例
:::::::::
COPY-FROM: paddle.nn.functional.rrelu:rrelu-example
