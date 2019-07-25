.. _cn_api_fluid_layers_distribute_fpn_proposals:

distribute_fpn_proposals
-------------------------------

.. py:function:: paddle.fluid.layers.distribute_fpn_proposals(fpn_rois, min_level, max_level, refer_level, refer_scale, name=None)

在 Feature Pyramid Networks（FPN）模型中，需要将所有proposal分配到不同的FPN级别，包括proposal的比例，引用比例和引用级别。 此外，为了恢复proposals的顺序，我们返回一个数组，该数组表示当前proposals中的原始RoIs索引。 要计算每个RoI的FPN级别，公式如下：

.. math::
    roi\_scale &= \sqrt{BBoxArea(fpn\_roi)}\\
    level = floor(&\log(\frac{roi\_scale}{refer\_scale}) + refer\_level)

其中BBoxArea方法用来计算每个RoI的区域。


参数：
    - **fpn_rois** （variable） - 输入fpn_rois，第二个维度为4。
    - **min_level** （int） - 产生proposal最低级别FPN层。
    - **max_level** （int） - 产生proposal最高级别FPN层。
    - **refer_level** （int） - 具有指定比例的FPN层的引用级别。
    - **refer_scale** （int） - 具有指定级别的FPN层的引用比例。
    - **name** （str | None） - 此算子的名称。

返回：返回一个元组（multi_rois，restore_ind）。 multi_rois是分段张量变量的列表。 restore_ind是具有形状[N，1]的2D张量，N是总rois的数量。 它用于恢复fpn_rois的顺序。

返回类型：   tuple


**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    fpn_rois = fluid.layers.data(
        name='data', shape=[4], dtype='float32', lod_level=1)
    multi_rois, restore_ind = fluid.layers.distribute_fpn_proposals(
        fpn_rois=fpn_rois,
        min_level=2,
        max_level=5,
        refer_level=4,
        refer_scale=224)



