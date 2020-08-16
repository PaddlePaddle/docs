l1_loss
-------------------------------

.. py:function:: paddle.nn.functional.l1_loss(x, label, reduction='mean', name=None)

该接口计算输入 ``x`` 和标签 ``label`` 间的 `L1 loss` 损失。

该损失函数的数学计算公式如下：

当 `reduction` 设置为 ``'none'`` 时，
    
    .. math::
        Out = \lvert x - label\rvert

当 `reduction` 设置为 ``'mean'`` 时，

    .. math::
       Out = MEAN(\lvert x - label\rvert)

当 `reduction` 设置为 ``'sum'`` 时，
    
    .. math::
       Out = SUM(\lvert x - label\rvert)


参数
:::::::::
    - **x** (Tensor): - 输入的Tensor，维度是[N, *], 其中N是batch size， `*` 是任意数量的额外维度。数据类型为：float32、float64、int32、int64。
    - **label** (Tensor): - 标签，维度是[N, *], 与 ``x`` 相同。数据类型为：float32、float64、int32、int64。
    - **reduction** (str, 可选): - 指定应用于输出结果的计算方式，可选值有: ``'none'``, ``'mean'``, ``'sum'`` 。默认为 ``'mean'``，计算 `L1Loss` 的均值；设置为 ``'sum'`` 时，计算 `L1Loss` 的总和；设置为 ``'none'`` 时，则返回 `L1Loss`。
    - **name** (str，可选): - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。

返回
:::::::::
``Tensor``, 输入 ``x`` 和标签 ``label`` 间的 `L1 loss` 损失。如果 :attr:`reduction` 是 ``'none'``, 则输出Loss的维度为 [N, *], 与输入 ``x`` 相同。如果 :attr:`reduction` 是 ``'mean'`` 或 ``'sum'``, 则输出Loss的维度为 [1]。


代码示例
:::::::::

.. code-block:: python

        import paddle
        import numpy as np
        
        paddle.disable_static()
        x_data = np.array([[1.5, 0.8], [0.2, 1.3]]).astype("float32")
        label_data = np.array([[1.7, 1], [0.4, 0.5]]).astype("float32")
        x = paddle.to_tensor(x_data)
        label = paddle.to_tensor(label_data)

        l1_loss = paddle.nn.functional.l1_loss(x, label)
        print(l1_loss.numpy())  
        # [0.35]

        l1_loss = paddle.nn.functional.l1_loss(x, label, reduction='none')
        print(l1_loss.numpy())  
        # [[0.20000005 0.19999999]
        # [0.2        0.79999995]]

        l1_loss = paddle.nn.functional.l1_loss(x, label, reduction='sum')
        print(l1_loss.numpy())  
        # [1.4]
