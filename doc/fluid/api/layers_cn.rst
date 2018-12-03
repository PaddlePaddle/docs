

.. _cn_api_fluid_layers_beam_search:

beam_search
>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.beam_search(pre_ids, pre_scores, ids, scores, beam_size, end_id, level=0, name=None)

在机器翻译任务中，束搜索(Beam search)是选择候选词的一种经典算法

更多细节参考 `Beam Search <https://en.wikipedia.org/wiki/Beam_search>`_

该层对束进行一次搜索。根据所有源句子的 ``scores `` , 从 ``ids`` 中选择当前步骤的 top-K 候选词的id，其中 ``K`` 是 ``beam_size`` 和 ``ids``， ``scores`` 是计算单元的预测结果。 另外， ``pre_id`` 和 ``pre_scores`` 是上一步中 ``beam_search`` 的输出，用于特殊处理结束边界。

注意，传入的 ``scores`` 应该是累积分数，并且，在计算累积分数之前应该使用额外的 operators 进行长度惩罚，也建议在计算前查找top-K，然后使用top-K候选项。

有关完全波束搜索用法示例，请参阅以下示例：
  
  fluid/tests/book/test_machine_translation.py
  
.. math::
            \\L2WeightDecay=reg\_coeff*parameter\\

参数:
  - **pre_ids** （Variable） -  LodTensor变量，它是上一步 beam_search的输出。在第一步中。它应该是LodTensor，shape为（batchsize，1）， lod [[0,1，...，batchsize]，[0,1，...，batchsize]]
  - **pre_scores** （Variable） -  LodTensor变量，它是上一步中beam_search的输出
  - **ids** （Variable） - 包含候选ID的LodTensor变量。shpae为（batchsize×beamize，K），其中 ``K`` 应该是 ``beam_size``
  - **score** （Variable） - 与 ``ids`` 及其shape对应的累积分数的LodTensor变量, 与 ``ids`` 的shape相同。
  - **beam_size** （int） - 束搜索中的束宽度。
  - **end_id** （int） - 结束标记的id。
  - **level** （int，default 0） - 可忽略，当前不能更改。它表示lod的级别，解释如下。 ``ids`` 的 lod 级别应为2.第一级是源级别， 描述每个源sentece（beam）的前缀（分支）的数量，第二级是描述这些候选者属于前缀的句子级别的方式。链接前缀和所选候选者的路径信息保存在lod中。
  - **name** （str | None） - 该层的名称（可选）。如果设置为None，则自动命名该层。

返回：LodTensor pair ， 包含所选的ID和相应的分数。  

**代码示例**

..  code-block:: python
    
    # Suppose `probs` contains predicted results from the computation
    # cell and `pre_ids` and `pre_scores` is the output of beam_search
    # at previous step.
    topk_scores, topk_indices = layers.topk(probs, k=beam_size)
    accu_scores = layers.elementwise_add(
                                          x=layers.log(x=topk_scores)),
                                          y=layers.reshape(
                                              pre_scores, shape=[-1]),
                                          axis=0)
    selected_ids, selected_scores = layers.beam_search(
                                          pre_ids=pre_ids,
                                          pre_scores=pre_scores,
                                          ids=topk_indices,
                                          scores=accu_scores,
                                          beam_size=beam_size,
                                          end_id=end_id)



.. _cn_api_fluid_layers_multiplex:

multiplex
>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.multiplex(inputs, index)

引用给定的索引变量，该层从输入变量中选择行构造Multiplex变量。 假设有m个输入变量，:math:`I_{i}` 代表第i个输入变量， :math:`i \in [0，m]` 。 所有输入变量都是具有相同形状的张量[d0，d1，...，dR]。 请注意，输入张量的等级应至少为2.每个输入变量将被视为形状为[M，N]的二维矩阵，其中M表示d0，N表示d1 * d2 * ... * dR。 设 :math:`I_{i}[j]` 为第i个输入变量的第j行。 给定的索引变量是具有形状[M，1]的2-D张量。 设ID[i]为索引变量的第i个索引值。 然后输出变量将是一个形状为[d0，d1，...，dR]的张量。 如果将输出张量视为具有形状[M，N]的2-D矩阵,并且令O[i]为矩阵的第i行，则O[i]等于 :math:`I_{ID}[i][i]` 
  
- Ids: 索引张量
- X[0 : N - 1]: 输出的候选张量度(N >= 2).
- 对于从0到batchSize  -  1的每个索引i，输出是（Ids [i]） -  th张量的第i行

对于第i行的输出张量：

.. math::
            \\y[i]=x_k[i]\\
            

参数:

  - **inputs** （list） - 要从中收集的变量列表。所有变量的形状相同，秩至少为2
  - **index** （Variable） -  Tensor <int32>，索引变量为二维张量，形状[M, 1]，其中M为批大小。

返回：multiplex 张量

**代码示例**

..  code-block:: python
    
   import paddle.fluid as fluid
   
   x1 = fluid.layers.data(name='x1', shape=[4], dtype='float32')
   x2 = fluid.layers.data(name='x2', shape=[4], dtype='float32')
   index = fluid.layers.data(name='index', shape=[1], dtype='int32')
   out = fluid.layers.multiplex(inputs=[x1, x2], index=index)
   

.. _cn_api_fluid_layers_layer_norm:

layer_norm
>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.layer_norm(input, scale=True, shift=True, begin_norm_axis=1, epsilon=1e-05, param_attr=None, bias_attr=None, act=None, name=None)

假设特征向量存在于维度 ``begin_norm_axis ... rank (input）`` 上，计算大小为 ``H`` 的特征向量a在该维度上的矩统计量，然后使用相应的统计量对每个特征向量进行归一化。 之后，如果设置了 ``scale`` 和 ``shift`` ，则在标准化的张量上应用可学习的增益和偏差以进行缩放和移位。

请参考 `Layer Normalization <https://arxiv.org/pdf/1607.06450v1.pdf>`_ 
            
公式如下

.. math::
            \\\mu=\frac{1}{H}\sum_{i=1}^{H}a_i\\
.. math::
            \\\sigma=\sqrt{\frac{1}{H}\sum_i^H{a_i-\mu}}\\
.. math::
             \\h=f(\frac{g}{\sigma}(a-\mu) + b)\\
             
- :math: `\alpha` : 该层神经元输入总和的向量表示
- :math:  `H` : 层中隐藏的神经元个数
- :math:  `g` : 可训练的缩放因子参数
- :math:  `b` : 可训练的bias


参数:
  - **input** （Variable） - 输入张量变量。
  - **scale** （bool） - 是否在归一化后学习自适应增益g。默认为True。
  - **shift** （bool） - 是否在归一化后学习自适应偏差b。默认为True。
  - **begin_norm_axis** （int） - ``begin_norm_axis`` 到 ``rank（input）`` 的维度执行规范化。默认1。
  - **epsilon** （float） - 添加到方差的很小的值，以防止除零。默认1e-05。
  - **param_attr** （ParamAttr | None） - 可学习增益g的参数属性。如果  ``scale`` 为False，则省略 ``param_attr`` 。如果 ``scale`` 为True且 ``param_attr`` 为None，则默认 ``ParamAttr`` 将作为比例。如果添加了 ``param_attr``， 则将其初始化为1。默认None。
  - **bias_attr** （ParamAttr | None） - 可学习偏差的参数属性b。如果 ``shift`` 为False，则省略 ``bias_attr`` 。如果 ``shift`` 为True且 ``param_attr`` 为None，则默认 ``ParamAttr`` 将作为偏差。如果添加了 ``bias_attr`` ，则将其初始化为0。默认None。
  - **act** （str） - 激活函数。默认 None
  - **name** （str） - 该层的名称， 可选的。默认为None，将自动生成唯一名称。

返回： 标准化后的结果   

**代码示例**

..  code-block:: python
    
   data = fluid.layers.data(name='data', shape=[3, 32, 32],
                                           dtype='float32')
   x = fluid.layers.layer_norm(input=data, begin_norm_axis=1)

.. _cn_api_fluid_layers_label_smooth:

label_smooth
>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.label_smooth(label, prior_dist=None, epsilon=0.1, dtype='float32', name=None)

标签平滑是一种对分类器层进行正则化的机制，称为标签平滑正则化(LSR)。


由于直接优化正确标签的对数似然可能会导致过拟合，降低模型的适应能力，因此提出了标签平滑的方法来鼓励模型不那么自信。标签平滑是一种对分类器层进行正则化的机制，称为标签平滑正则化(LSR)。


由于直接优化正确标签的对数似然可能会导致过拟合，降低模型的适应能力，因此提出了标签平滑的方法来鼓励模型不那么自信。 标签平滑使用自身和一些固定分布μ的加权和替真实标签y。对于k类，即

.. math::

            \\\bar{y_k} ~ =(1−ϵ)*yk +ϵ*μk\\

其中1-ε和ε分别是权重，ỹk是平滑标签。 通常μ 使用均匀分布

.. math::

\\\yk ~ =(1−ϵ)∗yk +ϵ∗μk\\\

在1−ϵ和ϵ权重分别和ỹk是平滑的标签。通常均匀分布用于μ。


查看更多关于标签平滑的细节 https://arxiv.org/abs/1512.00567

参数：
  - **label** （Variable） - 包含标签数据的输入变量。 标签数据应使用 one-hot 表示。
  - **prior_dist** （Variable） - 用于平滑标签的先验分布。 如果未提供，则使用均匀分布。 prior_dist的shape应为（1，class_num）。
  - **epsilon** （float） - 用于混合原始真实分布和固定分布的权重。
  - **dtype** （np.dtype | core.VarDesc.VarType | str） - 数据类型：float32，float_64，int等。
  - **name** （str | None） - 此图层的名称（可选）。 如果设置为None，则将自动命名图层。

返回：张量变量, 包含平滑后的标签。
         
**代码示例**

..  code-block:: python
    
    label = layers.data(name="label", shape=[1], dtype="float32")
    one_hot_label = layers.one_hot(input=label, depth=10)
    smooth_label = layers.label_smooth(
    label=one_hot_label, epsilon=0.1, dtype="float32")

.. _cn_api_fluid_layers_scatter:

scatter
>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.scatter(input, index, updates, name=None)

Scatter Layer

通过更新第一维度上指定选索引的输入来获得输出。

.. math::
            \\Out=XOut[Ids]=Updates\\


参数：
  - **input** （Variable） - 秩> = 1的源输入
  - **index** （Variable） - 秩= 1的索引输入。 它的dtype应该是int32或int64，因为它用作索引
  - **updates** （Variable） - scatter op的更新值
  - **name** （str | None） - 输出变量名称。 默认None

返回：张量变量, 与输入张量的shape相同

返回类型：output（Variable）

**代码示例**

..  code-block:: python
    
    output = fluid.layers.scatter(input, index, updates)


.. _cn_api_fluid_layers_detection_map:

detection_map
>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.detection_map(detect_res, label, class_num, background_label=0, overlap_threshold=0.3, evaluate_difficult=True, has_state=None, input_states=None, out_states=None, ap_version='integral')

通常步骤如下
1 根据 detection 的输入和标签计算 true positive 和 false positive
2 计算 map 值，支持‘11 point’ 和‘integral’ map 算法。详情参考：https://sanchom.wordpress.com/tag/average-precision/ https://arxiv.org/abs/1512.02325


参数
  - **detect_res** （LoDTensor）- 具有形状[M，6]的 2-D LoDTensor。 每行有 6 个值：[label，confidence，xmin，ymin，xmax，ymax]，M 是此 mini batch 中检测结果的总数。 对于每个实例，第一维中的偏移称为 LoD，偏移数为 N + 1，如果 LoD [i + 1] - LoD [i] == 0，则表示没有检测到数据
  - **label** （LoDTensor）- 2 维 LoDTensor 表示真实被标记的数据。 每行有 6 个值：[label，xmin，ymin，xmax，ymax，is_difficult]或 5 个值：[label，xmin，ymin，xmax，ymax]，其中 N 是此 mini batch 中的真实数据的总数。 对于每个实例，第一维中的偏移称为 LoD，偏移数为 N + 1，如果 LoD [i + 1] - LoD [i] == 0，则表示没有真实数据
  - **class_num** （int）- 类别号
  - **background_label** （int，defalut：0）- background_labe 的索引，默认忽略。如果设置为-1，则将考虑所有类别
  - **overlap_threshold** （float）- 检测输出和真实数据的下限 jaccard 重叠阈值
  - **evaluate_difficult** （bool，默认为 true）- 控制是够计算 difficult data
  - **has_state** （Tensor <int>）- 具有形状[1]的张量，0 表示忽略输入状态，其中包括 PosCount，TruePos，FalsePos
  - **input_states**  - 如果不为 None，它包含 3 个元素：1 pos_count（Tensor）一个形状为[Ncls，1]的张量，存储输入中每个类别的正例个数，Ncls 是输入中的类别数。此输入用于在执行多个小批量累积计算时传递先前小批量生成的 AccumPosCount。当输入（PosCount）为空时，不执行累积计算，仅计算当前小批量的结果。2. true_pos（LoDTensor）具有形状[Ntp，2]的 2-D LoDTensor，存储每个类的输入真正正例。此输入用于传递前一个小批量生成的 AccumTruePos 多个小批量累计计算进行。 。3. false_pos（LoDTensor）具有形状[Nfp，2]的 2-D LoDTensor，存储每个类的输入误报示例。此输入用于传递多个小批量时前一个小批量生成的 AccumFalsePos 累计计算进行。 。
  - **out_states**  - 如果不是 None，它包含 3 个元素。1. accum_pos_count（Tensor）具有形状[Ncls，1]的张量，存储每个类的正例数。它结合了输入输入（PosCount）和从输入（检测）和输入（标签）计算的正例计数。2. accum_true_pos（LoDTensor）具有形状[Ntp'，2]的 LoDTensor，存储每个类的真正正例。它结合了输入（TruePos）和从输入（检测）和输入（标签）计算的真实正例。3. accum_false_pos（LoDTensor）具有形状[Nfp'，2]的 LoDTensor，存储每个类的误报示例。它结合了输入（FalsePos）和从输入（检测）和输入（标签）计算的误报示例。
  - **ap_version**  - （字符串，默认'integral'）AP 算法类型，'integral'或'11point'
 
 返回：（Tensor）具有形状[1]的张量，mAP评估结果
 
 **代码示例**

..  code-block:: python

      detect_res = fluid.layers.data(
                                    name='detect_res',
                                    shape=[10, 6],
                                    append_batch_size=False,
                                    dtype='float32')
      label = fluid.layers.data(
                                name='label',
                                shape=[10, 6],
                                append_batch_size=False,
                                dtype='float32')

      map_out = fluid.layers.detection_map(detect_res, label, 21)

.. _cn_api_fluid_layers_rpn_target_assign:

rpn_target_assign
>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.rpn_target_assign(bbox_pred, cls_logits, anchor_box, anchor_var, gt_boxes, is_crowd, im_info, rpn_batch_size_per_im=256, rpn_straddle_thresh=0.0, rpn_fg_fraction=0.5, rpn_positive_overlap=0.7, rpn_negative_overlap=0.3, use_random=True)

 **在Faster-RCNN 中为 RPN 分配目标层** 

给定锚点和真实框之间的Iou ，该层为每个锚点分配分类和回归目标，这些目标标签用于训练RPN。分类目标是二进制类标签（是或不是目标对象）。根据Faster-RCNN的论文，正标签是两种锚：（i）具有最高IoU的锚/锚与真实框重叠，或（ii）具有高于rpn_positive_overlap的IoU的锚（0.7）与任何真实框重叠。请注意，单个真实框可以为多个锚点分配正标签。对于所有真实框，非正向锚是指其IoU比率低于rpn_negative_overlap（0.3）。既不是正面也不是负面的锚点对训练目标没有作用。回归目标是与正锚相关联的编码的真实框。

参数：
  - **bbox_pred** （Variable） - 具有形状[N，M，4]的3-D张量表示M个边界bbox的预测位置。 N是批量大小，每个边界框有四个坐标值，布局为[xmin，ymin，xmax，ymax]
  - **cls_logits** （Variable） - 具有形状[N，M，1]的3-D张量表示预测的置信度预测。 N是批量大小，1是前景和背景sigmoid，M是边界框的数量
  - **anchor_box** （Variable） - 具有形状[M，4]的2-D张量保持M个框，每个框表示为[xmin，ymin，xmax，ymax]，[xmin，ymin]是锚框的左上顶部坐标，如果输入是图像特征图，则它们接近坐标系的原点。 [xmax，ymax]是锚box的右下坐标
  - **anchor_var** （Variable） - 具有形状[M，4]的2-D张量保持锚的扩展方差
  - **gt_boxes** （Variable） - 真实boudding框（bbox）是具有形状[Ng，4]的2D LoDTensor，Ng是小批量输入的真实boundding box的总数
  - **is_crowd** （Variable） -  1-D LoDTensor，表示 ground-truth 是 crowd
  - **im_info** （Variable） - 形状为[N，3]的2-D LoDTensor。 N是批量大小
  - **is the height, width and scale.**   （3） - 
  - **rpn_batch_size_per_im** （int） - 每个image的RPN示例总数
  - **rpn_straddle_thresh** （float） - 通过straddle_thresh像素移除超出图像的RPN锚
  - **rpn_fg_fraction** （float） - 标记为foreground（即class> 0）的RoI minibatch的目标分数，第0类是background
  - **rpn_positive_overlap** （float） - 锚点和真实框之间所需的最小重叠（anchor >  box)）对是一个正例
  - **rpn_negative_overlap** （float） - 锚点和真实框之间允许的最大重叠（anchor >  box)）对是一个反面例子
  
返回：返回一个元组(predicted_scores, predicted_location, target_label, target_bbox)。predicted_scores和predicted_location是RPN的预测结果。target_label和target_bbox分别是ground truth。predicted_location是一个二维张量，其形状为[F, 4]， target_bbox的形状与predicted_location的形状相同，F为前景锚的个数。predicted_scores是一个二维张量，其形状为[F + B, 1]， target_label的形状与predicted_scores的形状相同，B是背景锚的个数，F和B取决于这个算子的输入。

返回类型： 元组（tuple）

 **代码示例**

..  code-block:: python
  
  bbox_pred = layers.data(name=’bbox_pred’, shape=[100, 4],
                                  append_batch_size=False, dtype=’float32’)
  cls_logits = layers.data(name=’cls_logits’, shape=[100, 1],
                                  append_batch_size=False, dtype=’float32’)
  anchor_box = layers.data(name=’anchor_box’, shape=[20, 4],
                            append_batch_size=False, dtype=’float32’)
                            gt_boxes = layers.data(name=’gt_boxes’, shape=[10, 4],
                            append_batch_size=False, dtype=’float32’)
  loc_pred, score_pred, loc_target, score_target =
                             fluid.layers.rpn_target_assign(bbox_pred=bbox_pred,
                              cls_logits=cls_logits, anchor_box=anchor_box, gt_boxes=gt_boxes)
