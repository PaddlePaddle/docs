.. _cn_api_paddle_nn_functional_dice_loss:

dice_loss
-------------------------------

.. py:function:: paddle.nn.functional.dice_loss(input, label, epsilon=1e-05, name=None)

比较预测结果跟标签之间的相似度，通常用于二值图像分割，即标签为二值，也可以做多标签的分割。

dice_loss 定义为：

.. math::
        dice\_loss &= 1- \frac{2 * intersection\_area}{total\_rea}\\
                   &= \frac{(total\_area−intersection\_area)−intersection\_area}{total\_area}\\
                   &= \frac{union\_area−intersection\_area}{total\_area}

参数
::::::::::::

    - **input** (Tensor) - 分类的预测概率，秩大于等于 2 的多维 Tensor，维度为 :math:`[N_1, N_2, ..., N_k, D]`。第一个维度的大小是 batch_size，最后一维的大小 D 是类别数目。数据类型是 float32 或者 float64。
    - **label** (Tensor)- 正确的标注数据(groud truth)，与输入 ``input`` 的秩相同的 Tensor，维度为 :math:`[N_1, N_2, ..., N_k, 1]`。第一个维度的大小是 batch_size，最后一个维度的大小是 1。数据类型为 int32 或者 int64。
    - **epsilon** (float，可选) - 将会加到分子和分母上的数，浮点型的数值。如果输入和标签都为空，则确保 dice 为 1。默认值：0.00001。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
 Tensor，shape 为 [batch_size, 1]，数据类型与 ``input`` 相同


代码示例
::::::::::::

COPY-FROM: paddle.nn.functional.dice_loss
