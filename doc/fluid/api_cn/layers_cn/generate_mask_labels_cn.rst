.. _cn_api_fluid_layers_generate_mask_labels:

generate_mask_labels
-------------------------------

.. py:function:: paddle.fluid.layers.generate_mask_labels(im_info, gt_classes, is_crowd, gt_segms, rois, labels_int32, num_classes, resolution)

**为Mask-RCNN生成mask标签**

对于给定的 RoI (Regions of Interest) 和相应的标签，该算子可以对前景RoI进行采样。 该mask branch对每个前景RoI还具有 :math:`K*M^{2}` 维输出目标，用于编码分辨率为M×M的K个二进制mask，K个种类中的各种类分别对应一个这样的二进制mask。 此mask输出目标用于计算掩码分支的损失。

请注意groud-truth（真实值，下简称GT）分段的数据格式。假设分段如下， 第一个实例有两个GT对象。 第二个实例有一个GT对象，该对象有两个GT分段。


::

    #[
    #  [[[229.14, 370.9, 229.14, 370.9, ...]],
    #   [[343.7, 139.85, 349.01, 138.46, ...]]], # 第0个实例对象
    #  [[[500.0, 390.62, ...],[115.48, 187.86, ...]]] # 第1个实例对象
    #]

    batch_masks = []
    for semgs in batch_semgs:
        gt_masks = []
        for semg in semgs:
            gt_segm = []
            for polys in semg:
                gt_segm.append(np.array(polys).reshape(-1, 2))
            gt_masks.append(gt_segm)
        batch_masks.append(gt_masks)


    place = fluid.CPUPlace()
    feeder = fluid.DataFeeder(place=place, feed_list=feeds)
    feeder.feed(batch_masks)


参数：
    - **im_info**  (Variable) – 具有形状[N，3]的2-D张量。 N是批量大小，其每个元素是图像的[高度，宽度，比例]，对应第二维中的3。图像比例是 :math:`\frac{target\_size}{original\_size}` 。
    - **gt_classes**  (Variable) – 形为[M，1]的2-D LoDTensor。 M是真实值的总数，其每个元素都是一个类标签，对应第二维中的1。
    - **is_crowd**  (Variable) – 一个形为 ``gt_classes`` 的2-D LoDTensor，每个元素都是一个标志，指示一个groundtruth是否为crowd（群）。
    - **gt_segms**  (Variable) – 这个输入是一个形状为[S，2]的2D LoDTensor，它的LoD级别为3。通常用户不需要理解LoD，但用户应该在Reader中返回正确的数据格式。LoD [0]表示每个实例中GT对象的数目。 LoD [1]表示每个对象的分段数。 LoD [2]表示每个分段的多边形(polygon)数。S为多边形坐标点的总数。每个元素是（x，y）坐标点。
    - **rois**  (Variable) – 形为[R，4]的2-D LoDTensor。 R是RoI的总数，其中每个元素是在原始图像范围内具有（xmin，ymin，xmax，ymax）格式的边界框(bounding box)。
    - **labels_int32**  (Variable) – 形为[R，1]且类型为int32的2-D LoDTensor。 R与rois中的R含义相同。每个元素都反映了RoI的一个类标签。
    - **num_classes**  (int) – 种类数目
    - **resolution**  (int) – mask预测的分辨率

返回：
    - 形为[P，4]的2D LoDTensor。 P是采样出的RoI总数。每个元素都是在原始图像大小范围内具有[xmin，ymin，xmax，ymax]格式的边界框(bounding box)。
    - mask_rois_has_mask_int32（Variable）：形状为[P，1]的2D LoDTensor，其中每个元素为对于输入的RoI进行输出的mask RoI 索引
    - mask_int32（Variable）：形状为[P，K * M * M]的2D LoDTensor，K为种类数，M为mask预测的分辨率，每个元素都是二进制目标mask值。

返回类型：mask_rois (Variable)

**代码示例**：

.. code-block:: python
    
    import paddle.fluid as fluid

    im_info = fluid.layers.data(name="im_info", shape=[3],
        dtype="float32")
    gt_classes = fluid.layers.data(name="gt_classes", shape=[1],
        dtype="float32", lod_level=1)
    is_crowd = fluid.layers.data(name="is_crowd", shape=[1],
        dtype="float32", lod_level=1)
    gt_masks = fluid.layers.data(name="gt_masks", shape=[2],
        dtype="float32", lod_level=3)
    # rois, roi_labels 可以是fluid.layers.generate_proposal_labels的输出
    rois = fluid.layers.data(name="rois", shape=[4],
        dtype="float32", lod_level=1)
    roi_labels = fluid.layers.data(name="roi_labels", shape=[1],
        dtype="int32", lod_level=1)
    mask_rois, mask_index, mask_int32 = fluid.layers.generate_mask_labels(
        im_info=im_info,
        gt_classes=gt_classes,
        is_crowd=is_crowd,
        gt_segms=gt_masks,
        rois=rois,
        labels_int32=roi_labels,
        num_classes=81,
        resolution=14)





