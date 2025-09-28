.. _cn_api_paddle_vision_ops_distribute_fpn_proposals:

distribute_fpn_proposals
-------------------------------

.. py:function:: paddle.vision.ops.distribute_fpn_proposals(fpn_rois, min_level, max_level, refer_level, refer_scale, pixel_offset=False, rois_num=None, name=None)



在 Feature Pyramid Networks（FPN）模型中，需要依据 proposal 的尺度和参考尺度与级别将所有 proposal 分配到不同的 FPN 级别中。此外，为了恢复 proposals 的顺序，我们返回一个数组，该数组表示当前 proposals 中的原始 RoIs 索引。计算每个 RoI 的 FPN 级别的公式如下：

.. math::
    roi\_scale &= \sqrt{BBoxArea(fpn\_roi)}\\
    level = floor(&\log(\frac{roi\_scale}{refer\_scale}) + refer\_level)

其中 BBoxArea 为用来计算每个 RoI 的区域的方法。


参数
::::::::::::

    - **fpn_rois** （Tensor） - 输入的 FPN RoIs。是形状为[N,4]的 2-D Tensor，其中 N 为检测框的个数，数据类型为 float32 或 float64。
    - **min_level** （int） - 产生 proposal 的最低级别 FPN 层。
    - **max_level** （int） - 产生 proposal 的最高级别 FPN 层。
    - **refer_level** （int） - 具有指定比例的 FPN 层的引用级别。
    - **refer_scale** （int） - 具有指定级别的 FPN 层的引用比例。
    - **pixel_offset** (bool, 可选）- 是否有像素偏移。如果是 True, 在计算形状大小时时会偏移 1。默认值为 False。
    - **rois_num** (Tensor, 可选): 每张图所包含的 RoI 数量。是形状为[B]的 1-D Tensor, 数据类型为 int32。其中 B 是图像数量。如果``rois_num`` 不为 None， 将会返回一个形状为[B]的 1-D Tensor, 其中每个元素是每张图在对应层级上的 RoI 数量。默认值为 None。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::

- **multi_rois** (List) - 长度为（max_level-min_level+1）的列表，其中元素为 Variable，维度为[M, 4]的 2-D Tensor，M 为每个级别 proposal 的个数，数据类型为 float32 或 float64。表示每个 FPN 级别包含的 proposals。
- **restore_ind** (Tensor) - 维度为[N，1]的 Tensor，N 是总 rois 的数量。数据类型为 int32。它用于恢复 fpn_rois 的顺序。
- **rois_num_per_level** (List) - 一个包含 1-D Tensor 的 List。其中每个元素是每张图在对应层级上的 RoI 数量。数据类型为 int32。

代码示例
::::::::::::

COPY-FROM: paddle.vision.ops.distribute_fpn_proposals
