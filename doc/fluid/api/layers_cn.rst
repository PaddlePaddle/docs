

.. _cn_api_fluid_layers_ssd_loss

ssd_loss
>>>>>>>>>>>>

.. py:class::  paddle.fluid.layers.ssd_loss(location, confidence, gt_box, gt_label, prior_box, prior_box_var=None, background_label=0, overlap_threshold=0.5, neg_pos_ratio=3.0, neg_overlap=0.5, loc_loss_weight=1.0, conf_loss_weight=1.0, match_type='per_prediction', mining_type='max_negative', normalize=True, sample_size=None)

Multi-box loss layer, 用于 SSD 目标检测算法

该层在给定位置偏移预测、置信度预测、先验框和真实 boudding boxes 和标签以及 Hard Example Mining，来计算 SSD 的 loss， 该 loss 是 localization loss（或 regression loss）和 confidence loss（或 classification loss）的加权和，步骤如下:
  
  1 用二部图匹配算法（bipartite matching algorithm）算出匹配的边框
    1.1 计算真实 box 和先验 box 之间的 IOU 相似度
    1.2 采用二部匹配算法计算匹配 boundding box

  2  计算 mining hard examples 的置信度
    2.1 根据匹配的索引获取目标标签
    2.2 计算 confidence loss

  3 应用 mining hard examples 得到负示例索引并更新匹配的索引
  
  4 分配分类和回归目标
    4.1 根据前面的框对 bbox 进行编码
    4.2 分配回归的目标
    4.3 指定分类的目标

  5 计算总体客观损失。
    5.1 计算 confidence loss
    5.2 计算 ocalization loss
    5.3 计算总体加权loss

  参数：

    - **location**（Variable） - 位置预测是 shape 为[N，Np，4]的 3D 张量，N 是 batch 的大小，Np 是每个实例的预测总数。4 是坐标值的个数，layout 为[xmin，ymin，xmax，ymax]。
    - **confidence**（Variable） - 置信度预测是一个三维张量，shape 为[N, Np, C]，N 和 Np 在位置上相同，C 是类别号
    - **gt_box**（Variable） - 真实 boudding bbox 是 2D LoDTensor，shape 为[Ng，4]，Ng 是 mini-batch 中的真实 bbox 的总数
    - **gt_label**（Variable） - 真实标签是一个二维 LoDTensor，shape 为[Ng，1]
    - **prior_box**（Variable） - 先验框是 2 维张量，shape 为[Np，4]
    - **prior_box_var**（Variable） - 先验框方差是具有 shape 为[Np，4]的 2 维张量。
    - **background_label**（int） - 背景标签的索引，默认为 0。
    - **overlap_threshold**（float） - 如果 match_type 为'per_prediction'，请使用 overlap_threshold 确定额外匹配的 bbox
            找到匹配的boxes。默认为 0.5。
    - **neg_pos_ratio**（float） - 负框与正框的比率，仅在 mining_type 为'max_negative'时使用，默认：3.0。
    - **neg_overlap**（float） - 非匹配预测的负重叠上限。仅当 mining_type 为'max_negative'时使用，默认为 0.5。
    - **loc_loss_weight**（float） - localization loss 的权重，默认为 1.0。
    - **conf_loss_weight**（float） - confidence loss 的权重，默认为 1.0。
    - **match_type**（str） - 训练期间匹配方法的类型应为'bipartite'或'per_prediction'，默认'per_prediction'。
    - **mining_type**（str） - hard example mining 类型，取之可以是'hard_example'或'max_negative'，目前只支持 max_negative。
    - **normalize**（bool） -  是否通过输出位置的总数对 SSD 损失进行规 normalization，默认为 True。
    - **sample_size**（int） - 负框的最大样本大小，仅在 mining_type 为'hard_example'时使用。

返回: localization loss 和 confidence loss 的加权和，形状为[N * Np, 1]，N 和 Np 相同。

抛出异常: ValueError 如果 mining_type 是' hard_example '，抛出 ValueError。现在只支持 mining type 的类型为 max_negative。


**代码示例**

.. code-block:: python

    pb = fluid.layers.data(
                    name='prior_box',
                    shape=[10, 4],
                    append_batch_size=False,
                    dtype='float32')
    pbv = fluid.layers.data(
                    name='prior_box_var',
                    shape=[10, 4],
                    append_batch_size=False,
                    dtype='float32')
    loc = fluid.layers.data(name='target_box', shape=[10, 4], dtype='float32')
    scores = fluid.layers.data(name='scores', shape=[10, 21], dtype='float32')
    gt_box = fluid.layers.data(
            name='gt_box', shape=[4], lod_level=1, dtype='float32')
    gt_label = fluid.layers.data(
            name='gt_label', shape=[1], lod_level=1, dtype='float32')
    loss = fluid.layers.ssd_loss(loc, scores, gt_box, gt_label, pb, pbv)


.. _cn_api_fluid_layers_polygon_box_transform

polygon_box_transform
>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.polygon_box_transform(input, name=None)  

PolygonBoxTransform 算子。

输入是检测网络的最终几何输出。我们使用 2*n 来表示从 polygon_box 中的 n 个点到像素位置的偏移。由于每个偏移包含两个数字(xi, yi)，所以何输出包含 2*n 个通道。

参数：
    - **input**（Variable） - shape 为[batch_size，geometry_channels，height，width]

返回：与输入 shpae 相同

返回类型：output（Variable）


.. _cn_api_fluid_layers_accuracy

accuracy
>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.accuracy(input, label, k=1, correct=None, total=None)

accuracy layer. 参考 https://en.wikipedia.org/wiki/Precision_and_recall

使用输入和标签计算准确率。 每个类别中top k 中正确预测的个数。Note：准确率的 dtype 由输入决定。 输入和标签 dtype 可以不同。

参数：
    - **input** (Variable)-该层的输入，即网络的预测。支持 Carry LoD。
    - **label** (Variable)-数据集的标签。
    - **k** (int) - 每个类别的 top k
    - **correct** (Variable)-正确的预测个数。
    - **total** (Variable)-总共的样本数。

返回:	正确率

返回类型:	变量（Variable）


