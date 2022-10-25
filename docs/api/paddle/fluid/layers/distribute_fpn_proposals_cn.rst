.. _cn_api_fluid_layers_distribute_fpn_proposals:

distribute_fpn_proposals
-------------------------------

.. py:function:: paddle.fluid.layers.distribute_fpn_proposals(fpn_rois, min_level, max_level, refer_level, refer_scale, name=None)




**该op仅支持LoDTensor输入**。在 Feature Pyramid Networks（FPN）模型中，需要依据proposal的尺度和参考尺度与级别将所有proposal分配到不同的FPN级别中。此外，为了恢复proposals的顺序，我们返回一个数组，该数组表示当前proposals中的原始RoIs索引。要计算每个RoI的FPN级别，公式如下：

.. math::
    roi\_scale &= \sqrt{BBoxArea(fpn\_roi)}\\
    level = floor(&\log(\frac{roi\_scale}{refer\_scale}) + refer\_level)

其中BBoxArea方法用来计算每个RoI的区域。


参数
::::::::::::

    - **fpn_rois** （Variable） - 维度为[N,4]的2-D LoDTensor，其中N为检测框的个数，数据类型为float32或float64。
    - **min_level** （int32） - 产生proposal最低级别FPN层。
    - **max_level** （int32） - 产生proposal最高级别FPN层。
    - **refer_level** （int32） - 具有指定比例的FPN层的引用级别。
    - **refer_scale** （int32） - 具有指定级别的FPN层的引用比例。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。 

返回
::::::::::::


        - multi_rois（List）- 长度为（max_level-min_level+1）的列表，其中元素为Variable，维度为[M, 4]的2-D LoDTensor，M为每个级别proposal的个数，数据类型为float32或float64。表示每个FPN级别包含的proposals。
        - restore_ind（Variable）- 维度为[N，1]的Tensor，N是总rois的数量。数据类型为int32。它用于恢复fpn_rois的顺序。


返回类型
::::::::::::
Tuple


代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.distribute_fpn_proposals