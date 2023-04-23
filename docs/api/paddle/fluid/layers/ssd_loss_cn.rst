.. _cn_api_fluid_layers_ssd_loss:

ssd_loss
-------------------------------

.. py:function:: paddle.fluid.layers.ssd_loss(location, confidence, gt_box, gt_label, prior_box, prior_box_var=None, background_label=0, overlap_threshold=0.5, neg_pos_ratio=3.0, neg_overlap=0.5, loc_loss_weight=1.0, conf_loss_weight=1.0, match_type='per_prediction', mining_type='max_negative', normalize=True, sample_size=None)




该 OP 用于 SSD 物体检测算法的多窗口损失层

该层用于计算 SSD 的损失，给定位置偏移预测，置信度预测，候选框和真实框标签，以及难样本挖掘的类型。通过执行以下步骤，返回的损失是本地化损失（或回归损失）和置信度损失（或分类损失）的加权和：

1、通过二分匹配算法查找匹配的边界框。

        1.1、计算真实框与先验框之间的 IOU 相似度。

        1.2、通过二分匹配算法计算匹配的边界框。

2、计算难分样本的置信度

        2.1、根据匹配的索引获取目标标签。

        2.2、计算置信度损失。

3、应用难样本挖掘算法来获取负样本索引并更新匹配的索引。

4、分配分类和回归目标

        4.1、根据生成的候选框 bbox 进行编码。

        4.2、分配回归目标。

        4.3、分配分类目标。

5、计算总体的物体损失。

        5.1 计算置信度(confidence)损失。

        5.1 计算回归(location)损失。

        5.3 计算总体加权损失。

参数
::::::::::::

        - **location** （Variable）- 位置预测，具有形状[N，Np，4]的 3D-Tensor，N 是 batch 大小，Np 是每个实例的预测总数。4 是坐标的维数，布局是[xmin，ymin，xmax，ymax]，xmin，ymin 代表 box 左上坐标，xmax，ymax 代表 box 右下坐标，数据类型为 float32 或 float64。
        - **confidence**  (Variable) - 置信度(分类)预测，具有形状[N，Np，C]的 3D-Tensor，N 是 batch 大小，Np 是每个实例的预测总数，C 是类别数量，数据类型为 float32 或 float64。
        - **gt_box** （Variable）- 真实框(bbox)，具有形状[Ng，4]的 2D LoDTensor，Ng 是 mini-batch 输入的真实框（bbox）的总数，4 是坐标的维数，布局是[xmin，ymin，xmax，ymax]，xmin，ymin 代表 box 左上坐标，xmax，ymax 代表 box 右下坐标，数据类型为 float32 或 float64。
        - **gt_label** （Variable）- ground-truth 标签，具有形状[Ng，1]的 2D LoDTensor,Ng 是 mini-batch 输入的真实框（bbox）的总数，1 表示类别号，数据类型为 float32 或 float64。
        - **prior_box** （Variable）- 检测网络生成的候选框，具有形状[Np，4]的 2D-Tensor，Np 是生成的候选框总数，4 是坐标的维数，布局是[xmin，ymin，xmax，ymax]，xmin，ymin 代表 box 左上坐标，xmax，ymax 代表 box 右下坐标，数据类型为 float32 或 float64。。
        - **prior_box_var** （Variable）- 候选框的方差，具有形状[Np，4]的 2DTensor，形状及数据类型同 ``prior_box`` 。
        - **background_label** （int）- background 标签的索引，默认为 0。
        - **overlap_threshold** （float）- 额外匹配的 bbox 阈值，当找到匹配的框，如果 ``match_type`` 为'per_prediction'，使用 ``overlap_threshold`` 确定额外匹配的 bbox。默认为 0.5。
        - **neg_pos_ratio** （float）- 负框与正框的比率，仅在 ``mining_type`` 为'max_negative'时使用，默认为 3.0。
        - **neg_overlap** （float）- 不匹配预测的负重叠上限。仅当 ``mining_type`` 为'max_negative'时使用，默认为 0.5。
        - **loc_loss_weight** （float）- 回归损失的权重，默认为 1.0。
        - **conf_loss_weight** （float）- 置信度损失的权重，默认为 1.0。
        - **match_type** （str）- 训练期间匹配方法的类型应为'bipartite'或'per_prediction'，默认为'per_prediction'。
        - **mining_type** （str）- 难样本挖掘类型，分为'hard_example'或'max_negative'，目前只支持'max_negative'。
        - **normalize** （bool）- 是否通过输出位置的总数将 SSD 损失标准化，默认为 True。
        - **sample_size** （int）- 负样本框的最大样本大小，仅在 ``mining_type`` 为'hard_example'时使用。

返回
::::::::::::
  Variable(Tensor)  定位损失和置信度损失的加权和，具有形状[N * Np，1], N 是 batch 大小，Np 是每个实例的预测总数，数据类型为 float32 或 float64。

抛出异常
::::::::::::
        ``ValueError`` - 如果 ``mining_type`` 是'hard_example'，目前只支持 ``max_negative`` 的挖掘类型。

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.ssd_loss
