.. _cn_api_fluid_layers_dice_loss:

dice_loss
-------------------------------

.. py:function:: paddle.nn.functional.dice_loss(input, label, epsilon=1e-05)




该OP用来比较预测结果跟标签之间的相似度，通常用于二值图像分割，即标签为二值，也可以做多标签的分割。

dice_loss定义为:

.. math::
        dice\_loss &= 1- \frac{2 * intersection\_area}{total\_rea}\\
                   &= \frac{(total\_area−intersection\_area)−intersection\_area}{total\_area}\\
                   &= \frac{union\_area−intersection\_area}{total\_area}

参数:
    - **input** (Tensor) - 分类的预测概率，秩大于等于2的多维Tensor，维度为 :math:`[N_1, N_2, ..., N_k, D]` 。第一个维度的大小是batch_size，最后一维的大小D是类别数目。数据类型是float32或者float64
    - **label** (Tensor)- 正确的标注数据(groud truth)，与输入 ``input`` 的秩相同的Tensor，维度为 :math:`[N_1, N_2, ..., N_k, 1]` 。第一个维度的大小是batch_size，最后一个维度的大小是1。数据类型为int32或者int64
    - **epsilon** (float，可选) - 将会加到分子和分母上的数，浮点型的数值。如果输入和标签都为空，则确保dice为1。默认值:0.00001

返回: Tensor，shape为[batch_size, 1]，数据类型与 ``input`` 相同


**代码示例**

..  code-block:: python

    import paddle
    import paddle.nn.functional as F
    x = paddle.randn((3,224,224,2))
    label = paddle.randint(high=2, shape=(3,224,224,1))
    predictions = F.softmax(x)
    loss = F.dice_loss(input=predictions, label=label)








