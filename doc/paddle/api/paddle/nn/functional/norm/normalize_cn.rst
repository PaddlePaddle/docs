normalize
-------------------------------

.. py:function:: paddle.nn.functional.normalize(x, p=2, axis=1, epsilon=1e-12, name=None)

该接口使用 :math:`L_p` 范数沿维度 ``axis`` 对 ``x`` 进行归一化。计算公式如下：

.. math::

    y = \frac{x}{ \max\left( \lvert \lvert x \rvert \rvert_p, epsilon\right) }

.. math::
    \lvert \lvert x \rvert \rvert_p = \left(\sum_i {\lvert x_i\rvert^p}  \right)^{1/p}

其中 :math:`\sum_i{\lvert x_i\rvert^p}` 沿维度 ``axis`` 进行计算。


参数
:::::::::
    - **x** (Tensor) - 输入可以是N-D Tensor。数据类型为：float32、float64。
    - **p** (float|int, 可选) - 范数公式中的指数值。默认值:2
    - **axis** (int, 可选）- 要进行归一化的轴。如果 ``x`` 是1-D Tensor，轴固定为0。如果 `axis < 0`，轴为 `x.ndim + axis`。-1表示最后一维。
    - **epsilon** (float，可选) - 添加到分母上的值以防止分母除0。默认值为1e-12。
    - **name** (str，可选) - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。

返回
:::::::::
``Tensor``, 输出的形状和数据类型和 ``x`` 相同。

抛出异常：
:::::::::
    - ``TypeError`` - 当参数  ``p`` 或者 ``axis`` 的类型不符合要求时。或者当参数 ``x`` 的类型或数据类型不符合要求时。

代码示例
:::::::::

.. code-block:: python

        import numpy as np
        import paddle
        import paddle.nn.functional as F

        paddle.disable_static()
        x = np.arange(6, dtype=np.float32).reshape(2,3)
        x = paddle.to_tensor(x)
        y = F.normalize(x)
        print(y.numpy())
        # [[0.         0.4472136  0.8944272 ]
        # [0.42426404 0.5656854  0.7071067 ]]

        y = F.normalize(x, p=1.5)
        print(y.numpy())
        # [[0.         0.40862012 0.81724024]
        # [0.35684016 0.4757869  0.5947336 ]]

        y = F.normalize(x, axis=0)
        print(y.numpy())
        # [[0.         0.24253564 0.37139067]
        # [1.         0.97014254 0.9284767 ]]
