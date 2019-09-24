.. _cn_api_fluid_layers_mean_iou:

mean_iou
-------------------------------

.. py:function:: paddle.fluid.layers.mean_iou(input, label, num_classes)

均值IOU（Mean  Intersection-Over-Union）是语义图像分割中的常用的评价指标之一，它首先计算每个类的IOU，然后计算类之间的平均值。IOU定义如下:

.. math::

    IOU = \frac{true\_positive}{true\_positive+false\_positive+false\_negative}

先得到类别的预测结果，然后从中计算均值-IOU。

参数:
    - **input** (Variable) - 分割类别预测结果，类型为int32或int64的多维Tensor。
    - **label** (Variable) - 真实label，类型为int32或int64的多维Tensor，它的shape与input相同。
    - **num_classes** (int32) - 类别数目。

返回: 
    - **mean_iou** (Variable): 类型为float32的1-D Tensor，shape为[1]， 均值IOU的计算结果。
    - **out_wrong** (Variable) : 类型为int32的1-D Tensor，shape为[num_classes]，代表每个类别中错误的个数。
    - **out_correct** (Variable) :类型为int32的1-D Tensor，shape为[num_classes]，代表每个类别中正确的个数。


**代码示例**

..  code-block:: python

   import paddle.fluid as fluid
   iou_shape = [32, 32]
   num_classes = 5
   predict = fluid.layers.data(name='predict', shape=iou_shape)
   label = fluid.layers.data(name='label', shape=iou_shape)
   mean_iou, out_wrong, out_correct = fluid.layers.mean_iou(predict, label, num_classes)

    
