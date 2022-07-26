.. _cn_api_fluid_layers_generate_mask_labels:

generate_mask_labels
-------------------------------

.. py:function:: paddle.fluid.layers.generate_mask_labels(im_info, gt_classes, is_crowd, gt_segms, rois, labels_int32, num_classes, resolution)




**为Mask-RCNN生成mask标签**

对于给定的 RoI (Regions of Interest) 和 输入ground truth的分类标签和分割的坐标标签，采样出前景RoI，并返回其在输入 ``rois`` 中索引位置，并对每个RoI生成 :math:`K*M^{2}` 的二值mask标签。K为类别个数，M是RoI特征图大小。这些输出目标一般用于计算mask分支的损失。

请注意分割groud-truth（真实标签，下简称GT)数据格式，这里要求分割标签为坐标信息，假如，第一个实例有两个GT对象。第二个实例有一个GT对象，该GT分割标签是两段(例如物体中间被隔开)，输入标签格式组织如下：


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


参数
::::::::::::

    - **im_info** (Variable) – 维度为[N，3]的2-D Tensor，数据类型为float32。 N是batch size，其每个元素是图像的高度、宽度、比例，比例是图片预处理时resize之后的大小和原始大小的比例 :math:`\frac{target\_size}{original\_size}` 。
    - **gt_classes**  (Variable) – 维度为[M，1]的2-D LoDTensor，数据类型为int32，LoD层数为1。 M是的groud-truth总数，其每个元素是类别索引。
    - **is_crowd**  (Variable) – 维度和 ``gt_classes`` 相同的 LoDTensor，数据类型为int32，每个元素指示一个ground-truth是否为crowd（crowd表示一组对象的集合）。
    - **gt_segms**  (Variable) – 维度为[S，2]的2D LoDTensor，它的LoD层数为3，数据类型为float32。通常用户不需要理解LoD，但用户应该在Reader中返回正确的数据格式。LoD[0]表示每个实例中GT对象的数目。LoD[1]表示每个GT对象的标签分段数。LoD[2]表示每个分段标签多边形(polygon)坐标点的个数。S为所有示例的标签的多边形坐标点的总数。每个元素是（x，y）坐标点。
    - **rois**  (Variable) – 维度维度[R，4]的2-D LoDTensor，LoD层数为1，数据类型为float32。 R是RoI的总数，其中每个元素是在原始图像范围内具有（xmin，ymin，xmax，ymax）格式的bounding box。
    - **labels_int32**  (Variable) – 形为[R，1]且类型为int32的2-D LoDTensor，数据类型为int32。 R与 ``rois`` 中的R含义相同。每个元素表示RoI框的一个类别标签索引。
    - **num_classes**  (int) – 类别数目。
    - **resolution**  (int) – 特征图分辨率大小。

返回
::::::::::::

    - mask_rois (Variable)：维度为[P，4]的2-D LoDTensor，数据类型为float32。P是采样出的RoI总数，每个元素都是在原始图像大小范围内具有[xmin，ymin，xmax，ymax]格式的bounding box。
    - mask_rois_has_mask_int32（Variable)：维度为[P，1]的2-D LoDTensor，数据类型为int32。每个元素表示采样出的RoI在输入 ``rois`` 内的位置索引。
    - mask_int32（Variable)：维度为[P，K * M * M]的2-D LoDTensor，数据类型为int32。K为种类数，M为特征图的分辨率大小，每个元素都是二值mask标签。

返回类型
::::::::::::
tuple(Variable)

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.generate_mask_labels