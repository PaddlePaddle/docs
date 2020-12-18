smooth_l1_loss
-------------------------------

.. py:function:: paddle.nn.functional.smooth_l1_loss(input, label, reduction='mean', delta=1.0, name=None)

该OP计算输入input和标签label间的SmoothL1损失，如果逐个元素的绝对误差低于1，则创建使用平方项的条件
，否则为L1损失。在某些情况下，它可以防止爆炸梯度, 也称为Huber损失,该损失函数的数学计算公式如下：

    .. math::
         loss(x,y) = \frac{1}{n}\sum_{i}z_i

`z_i`的计算公式如下：

    .. math::

        \mathop{z_i} = \left\{\begin{array}{rcl}
        0.5(x_i - y_i)^2 & & {if |x_i - y_i| < delta} \\
        delta * |x_i - y_i| - 0.5 * delta^2 & & {otherwise}
        \end{array} \right.

参数
::::::::::
    - **input** (Tensor): 输入 `Tensor`， 数据类型为float32。其形状为 :math:`[N, C]` , 其中 `C` 为类别数。对于多维度的情形下，它的形状为 :math:`[N, C, d_1, d_2, ..., d_k]`，k >= 1。
    - **label** (Tensor): 输入input对应的标签值，数据类型为float32。数据类型和input相同。
    - **reduction** (string, 可选): - 指定应用于输出结果的计算方式，数据类型为string，可选值有: `none`, `mean`, `sum` 。默认为 `mean` ，计算`mini-batch` loss均值。设置为 `sum` 时，计算 `mini-batch` loss的总和。设置为 `none` 时，则返回loss Tensor。
    - **delta** (string, 可选): SmoothL1Loss损失的阈值参数，用于控制Huber损失对线性误差或平方误差的侧重。数据类型为float32。 默认值= 1.0。
    - **name** （string，可选） - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。



返回：返回计算 `smooth_l1_loss` 后的损失值。

返回类型：Tensor

**代码示例**

..  code-block:: python

            import paddle
            import numpy as np

            input = np.random.rand(3,3).astype("float32")
            label = np.random.rand(3,3).astype("float32")
            input = paddle.to_tensor(input_data)
            label = paddle.to_tensor(label_data)
            output = paddle.nn.functional.smooth_l1_loss(input,label)
            print(output)
