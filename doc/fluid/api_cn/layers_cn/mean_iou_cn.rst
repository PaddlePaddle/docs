.. _cn_api_fluid_layers_mean_iou:

mean_iou
-------------------------------

.. py:function:: paddle.fluid.layers.mean_iou(input, label, num_classes)

均值IOU（Mean  Intersection-Over-Union）是语义图像分割中的常用的评价指标之一，它首先计算每个语义类的IOU，然后计算类之间的平均值。定义如下:

.. math::

    IOU = \frac{true\_positive}{true\_positive+false\_positive+false\_negative}

在一个confusion矩阵中累积得到预测值，然后从中计算均值-IOU。

参数:
    - **input** (Variable) - 类型为int32或int64的语义标签的预测结果张量。
    - **label** (Variable) - int32或int64类型的真实label张量。它的shape应该与输入相同。
    - **num_classes** (int) - 标签可能的类别数目。

返回: 返回三个变量:

- mean_iou: 张量，形为[1]， 代表均值IOU。
- out_wrong: 张量，形为[num_classes]。每个类别中错误的个数。
- out_correct:张量，形为[num_classes]。每个类别中的正确的个数。

返回类型:   mean_iou (Variable),out_wrong(Variable),out_correct(Variable)

**代码示例**

..  code-block:: python

   import paddle.fluid as fluid
   predict = fluid.layers.data(name='predict', shape=[3, 32, 32])
   label = fluid.layers.data(name='label', shape=[1])
   iou, wrongs, corrects = fluid.layers.mean_iou(predict, label, num_classes)









