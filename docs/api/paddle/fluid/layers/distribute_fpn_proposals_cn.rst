.. _cn_api_fluid_layers_distribute_fpn_proposals:

distribute_fpn_proposals
-------------------------------

.. py:function:: paddle.fluid.layers.distribute_fpn_proposals(fpn_rois, min_level, max_level, refer_level, refer_scale, name=None)




**该 op 仅支持 LoDTensor 输入**。在 Feature Pyramid Networks（FPN）模型中，需要依据 proposal 的尺度和参考尺度与级别将所有 proposal 分配到不同的 FPN 级别中。此外，为了恢复 proposals 的顺序，我们返回一个数组，该数组表示当前 proposals 中的原始 RoIs 索引。要计算每个 RoI 的 FPN 级别，公式如下：

.. math::
    roi\_scale &= \sqrt{BBoxArea(fpn\_roi)}\\
    level = floor(&\log(\frac{roi\_scale}{refer\_scale}) + refer\_level)

其中 BBoxArea 方法用来计算每个 RoI 的区域。


参数
::::::::::::

    - **fpn_rois** （Variable） - 维度为[N,4]的 2-D LoDTensor，其中 N 为检测框的个数，数据类型为 float32 或 float64。
    - **min_level** （int32） - 产生 proposal 最低级别 FPN 层。
    - **max_level** （int32） - 产生 proposal 最高级别 FPN 层。
    - **refer_level** （int32） - 具有指定比例的 FPN 层的引用级别。
    - **refer_scale** （int32） - 具有指定级别的 FPN 层的引用比例。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::


        - multi_rois（List）- 长度为（max_level-min_level+1）的列表，其中元素为 Variable，维度为[M, 4]的 2-D LoDTensor，M 为每个级别 proposal 的个数，数据类型为 float32 或 float64。表示每个 FPN 级别包含的 proposals。
        - restore_ind（Variable）- 维度为[N，1]的 Tensor，N 是总 rois 的数量。数据类型为 int32。它用于恢复 fpn_rois 的顺序。


返回类型
::::::::::::
Tuple


代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.distribute_fpn_proposals
