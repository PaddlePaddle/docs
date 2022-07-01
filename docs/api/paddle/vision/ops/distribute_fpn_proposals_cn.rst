.. _cn_api_paddle_vision_ops_distribute_fpn_proposals:

distribute_fpn_proposals
-------------------------------

.. py:function:: paddle.vision.ops.distribute_fpn_proposals(fpn_rois, min_level, max_level, refer_level, refer_scale, pixel_offset=False, rois_num=None, name=None)



在 Feature Pyramid Networks（FPN）模型中，需要依据proposal的尺度和参考尺度与级别将所有proposal分配到不同的FPN级别中。此外，为了恢复proposals的顺序，我们返回一个数组，该数组表示当前proposals中的原始RoIs索引。计算每个RoI的FPN级别的公式如下：

.. math::
    roi\_scale &= \sqrt{BBoxArea(fpn\_roi)}\\
    level = floor(&\log(\frac{roi\_scale}{refer\_scale}) + refer\_level)

其中BBoxArea为用来计算每个RoI的区域的方法。


参数
::::::::::::

    - **fpn_rois** （Tensor） - 输入的FPN RoIs。是形状为[N,4]的2-D Tensor，其中N为检测框的个数，数据类型为float32或float64。
    - **min_level** （int） - 产生proposal的最低级别FPN层。
    - **max_level** （int） - 产生proposal的最高级别FPN层。
    - **refer_level** （int） - 具有指定比例的FPN层的引用级别。
    - **refer_scale** （int） - 具有指定级别的FPN层的引用比例。
    - **pixel_offset** (bool, 可选）- 是否有像素偏移。如果是True, 在计算形状大小时时会偏移1。默认值为False。
    - **rois_num** (Tensor, 可选): 每张图所包含的RoI数量。是形状为[B]的1-D Tensor, 数据类型为int32。其中B是图像数量。如果``rois_num`` 不为None， 将会返回一个形状为[B]的1-D Tensor, 其中每个元素是每张图在对应层级上的RoI数量。默认值为None。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。 

返回
::::::::::::

- **multi_rois**（List）- 长度为（max_level-min_level+1）的列表，其中元素为Variable，维度为[M, 4]的2-D LoDTensor，M为每个级别proposal的个数，数据类型为float32或float64。表示每个FPN级别包含的proposals。
- **restore_ind**（Tensor）- 维度为[N，1]的Tensor，N是总rois的数量。数据类型为int32。它用于恢复fpn_rois的顺序。
- **rois_num_per_level** (List) : 一个包含1-D Tensor的List。其中每个元素是每张图在对应层级上的RoI数量。数据类型为int32。

代码示例
::::::::::::

COPY-FROM: paddle.vision.ops.distribute_fpn_proposals
