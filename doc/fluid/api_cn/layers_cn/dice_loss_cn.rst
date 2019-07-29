.. _cn_api_fluid_layers_dice_loss:

dice_loss
-------------------------------

.. py:function:: paddle.fluid.layers.dice_loss(input, label, epsilon=1e-05)

dice_loss是比较两批数据相似度，通常用于二值图像分割，即标签为二值。

dice_loss定义为:

.. math::
        dice\_loss &= 1- \frac{2 * intersection\_area}{total\_rea}\\
                   &= \frac{(total\_area−intersection\_area)−intersection\_area}{total\_area}\\
                   &= \frac{union\_area−intersection\_area}{total\_area}

参数:
    - **input** (Variable) - rank>=2的预测。第一个维度是batch大小，最后一个维度是类编号。
    - **label** （Variable）- 与输入tensor rank相同的正确的标注数据（groud truth）。第一个维度是batch大小，最后一个维度是1。
    - **epsilon** (float) - 将会加到分子和分母上。如果输入和标签都为空，则确保dice为1。默认值:0.00001

返回: dice_loss shape为[1]。

返回类型:  dice_loss(Variable)

**代码示例**

..  code-block:: python

    import paddle.fluid as fluid
    x = fluid.layers.data(name='data', shape = [3, 224, 224, 2], dtype='float32')
    label = fluid.layers.data(name='label', shape=[3, 224, 224, 1], dtype='float32')
    predictions = fluid.layers.softmax(x)
    loss = fluid.layers.dice_loss(input=predictions, label=label)











