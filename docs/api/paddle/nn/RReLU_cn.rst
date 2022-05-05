.. _cn_api_nn_RReLU:

RReLU
-------------------------------
.. py:class:: paddle.nn.RReLU(lower=1./8., upper=1./3., name=None)

RReLU激活层（RReLU Activation Operator）。计算公式如下：

如果使用近似计算：

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
    - name (str, 可选) - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。

形状:
::::::::::
    - input: 任意形状的Tensor，默认数据类型为float32。
    - output: 和input具有相同形状的Tensor。

代码示例
:::::::::
COPY-FROM: paddle.nn.RRelu:RRelu-example
