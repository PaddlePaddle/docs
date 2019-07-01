==============
detection
==============


.. _cn_api_fluid_layers_anchor_generator:

anchor_generator
-------------------------------

.. py:function:: paddle.fluid.layers.anchor_generator(input, anchor_sizes=None, aspect_ratios=None, variance=[0.1, 0.1, 0.2, 0.2], stride=None, offset=0.5, name=None)

**Anchor generator operator**

为Faster RCNN算法生成anchor，输入的每一位产生N个anchor，N=size(anchor_sizes)*size(aspect_ratios)。生成anchor的顺序首先是aspect_ratios循环，然后是anchor_sizes循环。

参数：
    - **input** (Variable) - 输入特征图，格式为NCHW
    - **anchor_sizes** (list|tuple|float) - 生成anchor的anchor大小，以绝对像素的形式表示，例如：[64.,128.,256.,512.]。若anchor的大小为64，则意味着这个anchor的面积等于64**2。
    - **aspect_ratios** (list|tuple|float) - 生成anchor的高宽比，例如[0.5,1.0,2.0]
    - **variance** (list|tuple) - 变量，在框回归delta中使用。默认：[0.1,0.1,0.2,0.2]
    - **stride** (list|tuple) - anchor在宽度和高度方向上的步长，比如[16.0,16.0]
    - **offset** (float) - 先验框的中心位移。默认：0.5
    - **name** (str) - 先验框操作符名称。默认：None

返回：
    - Anchors(Varibale): 输出anchor，布局[H,W,num_anchors,4] , ``H``  是输入的高度， ``W`` 是输入的宽度， ``num_priors`` 是输入每位的框数,每个anchor格式（未归一化）为(xmin,ymin,xmax,ymax)

    - Variances(Variable): anchor的扩展变量布局为 [H,W,num_priors,4]。 ``H`` 是输入的高度， ``W`` 是输入的宽度， ``num_priors`` 是输入每个位置的框数,每个变量的格式为(xcenter,ycenter,w,h)。

返回类型：Anchors(Variable),Variances(Variable)

**代码示例**：

.. code-block:: python

    conv1 = fluid.layers.data(name='conv1', shape=[48, 16, 16], dtype='float32')
    anchor, var = fluid.layers.anchor_generator(
    input=conv1,
    anchor_sizes=[64, 128, 256, 512],
    aspect_ratios=[0.5, 1.0, 2.0],
    variance=[0.1, 0.1, 0.2, 0.2],
    stride=[16.0, 16.0],
    offset=0.5)









.. _cn_api_fluid_layers_bipartite_match:

bipartite_match
-------------------------------

.. py:function:: paddle.fluid.layers.bipartite_match(dist_matrix, match_type=None, dist_threshold=None, name=None)

该算子实现了贪心二分匹配算法，该算法用于根据输入距离矩阵获得与最大距离的匹配。对于输入二维矩阵，二分匹配算法可以找到每一行的匹配列（匹配意味着最大距离），也可以找到每列的匹配行。此算子仅计算列到行的匹配索引。对于每个实例，匹配索引的数量是输入距离矩阵的列号。

它有两个输出，匹配的索引和距离。简单的描述是该算法将最佳（最大距离）行实体与列实体匹配，并且匹配的索引在ColToRowMatchIndices的每一行中不重复。如果列实体与任何行实体不匹配，则ColToRowMatchIndices设置为-1。

注意：输入距离矩阵可以是LoDTensor（带有LoD）或Tensor。如果LoDTensor带有LoD，则ColToRowMatchIndices的高度是批量大小。如果是Tensor，则ColToRowMatchIndices的高度为1。

注意：此API是一个非常低级别的API。它由 ``ssd_loss`` 层使用。请考虑使用 ``ssd_loss`` 。

参数：
                - **dist_matrix** （变量）- 该输入是具有形状[K，M]的2-D LoDTensor。它是由每行和每列来表示实体之间的成对距离矩阵。例如，假设一个实体是具有形状[K]的A，另一个实体是具有形状[M]的B. dist_matrix [i] [j]是A[i]和B[j]之间的距离。距离越大，匹配越好。

                注意：此张量可以包含LoD信息以表示一批输入。该批次的一个实例可以包含不同数量的实体。

                - **match_type** （string | None）- 匹配方法的类型，应为'bipartite'或'per_prediction'。[默认'二分']。
                - **dist_threshold** （float | None）- 如果match_type为'per_prediction'，则此阈值用于根据最大距离确定额外匹配的bbox，默认值为0.5。

返回：        返回一个包含两个元素的元组。第一个是匹配的索引（matched_indices），第二个是匹配的距离（matched_distance）。

         **matched_indices** 是一个2-D Tensor，int类型的形状为[N，M]。 N是批量大小。如果match_indices[i][j]为-1，则表示B[j]与第i个实例中的任何实体都不匹配。否则，这意味着在第i个实例中B[j]与行match_indices[i][j]匹配。第i个实例的行号保存在match_indices[i][j]中。

         **matched_distance** 是一个2-D Tensor，浮点型的形状为[N，M]。 N是批量大小。如果match_indices[i][j]为-1，则match_distance[i][j]也为-1.0。否则，假设match_distance[i][j]=d，并且每个实例的行偏移称为LoD。然后match_distance[i][j]=dist_matrix[d]+ LoD[i]][j]。

返回类型：        元组(tuple)

**代码示例**

..  code-block:: python

         x = fluid.layers.data(name='x', shape=[4], dtype='float32')
         y = fluid.layers.data(name='y', shape=[4], dtype='float32')
         iou = fluid.layers.iou_similarity(x=x, y=y)
         matched_indices, matched_dist = fluid.layers.bipartite_match(iou)



.. _cn_api_fluid_layers_box_clip:

box_clip
-------------------------------

.. py:function:: paddle.fluid.layers.box_clip(input, im_info, name=None)

将box框剪切为 ``im_info`` 给出的大小。对于每个输入框，公式如下：

::

    xmin = max(min(xmin, im_w - 1), 0)
    ymin = max(min(ymin, im_h - 1), 0)
    xmax = max(min(xmax, im_w - 1), 0)
    ymax = max(min(ymax, im_h - 1), 0)

其中im_w和im_h是从im_info计算的：

::

    im_h = round(height / scale)
    im_w = round(weight / scale)


参数：
    - **input (variable)**  – 输入框，最后一个维度为4
    - **im_info (variable)**  – 具有（高度height，宽度width，比例scale）排列的形为[N，3]的图像的信息。高度和宽度是输入大小，比例是输入大小和原始大小的比率
    - **name (str)**  – 该层的名称。 为可选项

返回：剪切后的tensor

返回类型： Variable


**代码示例**

..  code-block:: python

    boxes = fluid.layers.data(
        name='boxes', shape=[8, 4], dtype='float32', lod_level=1)
    im_info = fluid.layers.data(name='im_info', shape=[3])
    out = fluid.layers.box_clip(
        input=boxes, im_info=im_info, inplace=True)










.. _cn_api_fluid_layers_box_coder:

box_coder
-------------------------------

.. py:function:: paddle.fluid.layers.box_coder(prior_box, prior_box_var, target_box, code_type='encode_center_size', box_normalized=True, name=None, axis=0)

Bounding Box Coder

编码/解码带有先验框信息的目标边界框

编码规则描述如下：

.. math::

    ox &= (tx - px)/pw/pxv

    oy &= (ty - py)/ph/pyv

    ow &= log(abs(tw/pw))/pwv

    oh &= log(abs(th/ph))/phv

解码规则描述如下：

.. math::

    ox &= (pw * pxv * tx * + px ) - tw/2

    oy &= (ph * pyv * ty * + py ) - th/2

    ow &= exp(pwv * tw ) * pw + tw/2

    oh &= exp(phv * th ) * ph + th/2

其中tx，ty，tw，th分别表示目标框的中心坐标、宽度和高度。同样地，px，py，pw，ph表示先验框地中心坐标、宽度和高度。pxv，pyv，pwv，phv表示先验框变量，ox，oy，ow，oh表示编码/解码坐标、宽度和高度。


在Box Decoding期间，支持两种broadcast模式。 假设目标框具有形状[N，M，4]，并且prior框的形状可以是[N，4]或[M，4]。 然后，prior框将沿指定的轴broadcast到目标框。


参数：
    - **prior_box** (Variable) - 张量，默认float类型的张量。先验框是二维张量，维度为[M,4]，存储M个框，每个框代表[xmin，ymin，xmax，ymax]，[xmin，ymin]是先验框的左顶点坐标，如果输入数图像特征图，则接近坐标原点。[xmax,ymax]是先验框的右底点坐标
    - **prior_box_var** (Variable|list|None) - 支持两种输入类型，一是二维张量，维度为[M,4]，存储M个prior box。另外是一个含有4个元素的list，所有prior box共用这个list。
    - **target_box** (Variable) - LoDTensor或者Tensor，当code_type为‘encode_center_size’，输入可以是二维LoDTensor，维度为[N,4]。当code_type为‘decode_center_size’输入可以为三维张量，维度为[N,M,4]。每个框代表[xmin,ymin,xmax,ymax]，[xmin,ymin]是先验框的左顶点坐标，如果输入数图像特征图，则接近坐标原点。[xmax,ymax]是先验框的右底点坐标。该张量包含LoD信息，代表一批输入。批的一个实例可以包含不同的实体数。
    - **code_type** (string，默认encode_center_size) - 编码类型用目标框，可以是encode_center_size或decode_center_size
    - **box_normalized** (boolean，默认true) - 是否将先验框作为正则框
    - **name**  (string) – box编码器的名称
    - **axis**  (int) – 在PriorBox中为axis指定的轴broadcast以进行框解码，例如，如果axis为0且TargetBox具有形状[N，M，4]且PriorBox具有形状[M，4]，则PriorBox将broadcast到[N，M，4]用于解码。 它仅在code_type为decode_center_size时有效。 默认设置为0。


返回：

       - ``code_type`` 为 ``‘encode_center_size’`` 时，形为[N,M,4]的输出张量代表N目标框的结果，目标框用M先验框和变量编码。
       - ``code_type`` 为 ``‘decode_center_size’`` 时，N代表batch大小，M代表解码框数

返回类型：output_box（Variable）



**代码示例**

.. code-block:: python

    prior_box = fluid.layers.data(name='prior_box',
                                  shape=[512, 4],
                                  dtype='float32',
                                  append_batch_size=False)
    target_box = fluid.layers.data(name='target_box',
                                   shape=[512,81,4],
                                   dtype='float32',
                                   append_batch_size=False)
    output = fluid.layers.box_coder(prior_box=prior_box,
                                    prior_box_var=[0.1,0.1,0.2,0.2],
                                    target_box=target_box,
                                    code_type="decode_center_size",
                                    box_normalized=False,
                                    axis=1)




.. _cn_api_fluid_layers_box_decoder_and_assign:

box_decoder_and_assign
-------------------------------

.. py:function:: paddle.fluid.layers.box_decoder_and_assign(prior_box, prior_box_var, target_box, box_score, box_clip, name=None)

边界框编码器。

根据prior_box来解码目标边界框。

解码方案为：

.. math::

    ox &= (pw \times pxv \times tx + px) - \frac{tw}{2}\\
    oy &= (ph \times pyv \times ty + py) - \frac{th}{2}\\
    ow &= \exp (pwv \times tw) \times pw + \frac{tw}{2}\\
    oh &= \exp (phv \times th) \times ph + \frac{th}{2}

其中tx，ty，tw，th分别表示目标框的中心坐标，宽度和高度。 类似地，px，py，pw，ph表示prior_box（anchor）的中心坐标，宽度和高度。 pxv，pyv，pwv，phv表示prior_box的variance，ox，oy，ow，oh表示decode_box中的解码坐标，宽度和高度。

box decode过程得出decode_box，然后分配方案如下所述：

对于每个prior_box，使用最佳non-background（非背景）类的解码值来更新prior_box位置并获取output_assign_box。 因此，output_assign_box的形状与PriorBox相同。




参数：
   - **prior_box** （Variable） - （Tensor，默认Tensor <float>）框列表PriorBox是一个二维张量，形状为[N，4]，它包含N个框，每个框表示为[xmin，ymin，xmax，ymax]， [xmin，ymin]是anchor框的左上坐标，如果输入是图像特征图，则它们接近坐标系的原点。 [xmax，ymax]是anchor框的右下坐标
   - **prior_box_var** （Variable） - （Tensor，默认Tensor <float>，可选）PriorBoxVar是一个二维张量，形状为[N，4]，它包含N组variance。 PriorBoxVar默认将所有元素设置为1
   - **target_box** （Variable） - （LoDTensor或Tensor）此输入可以是形状为[N，classnum * 4]的2-D LoDTensor。它拥有N个框的N个目标
   - **box_score** （变量） - （LoDTensor或Tensor）此输入可以是具有形状[N，classnum]的2-D LoDTensor，每个框表示为[classnum]，其中含有各分类概率值
   - **box_clip** （FLOAT） - （float，默认4.135，np.log（1000. / 16.））裁剪框以防止溢出
   - **name** （str | None） - 此算子的自定义名称


返回：两个变量：

     - decode_box（Variable）:( LoDTensor或Tensor）op的输出张量，形为[N，classnum * 4]，表示用M个prior_box解码的N个目标框的结果，以及每个类上的variance
     - output_assign_box（Variable）:( LoDTensor或Tensor）op的输出张量，形为[N，4]，表示使用M个prior_box解码的N个目标框的结果和BoxScore的最佳非背景类的方差

返回类型：   decode_box(Variable), output_assign_box(Variable)


**代码示例**

.. code-block:: python

    pb = fluid.layers.data(
        name='prior_box', shape=[4], dtype='float32')
    pbv = fluid.layers.data(
        name='prior_box_var', shape=[4], dtype='float32', append_batch_size=False))
    loc = fluid.layers.data(
        name='target_box', shape=[4*81], dtype='float32')
    scores = fluid.layers.data(
        name='scores', shape=[81], dtype='float32')
    decoded_box, output_assign_box = fluid.layers.box_decoder_and_assign(
        pb, pbv, loc, scores, 4.135)




.. _cn_api_fluid_layers_collect_fpn_proposals:

collect_fpn_proposals
-------------------------------

.. py:function:: paddle.fluid.layers.collect_fpn_proposals(multi_rois, multi_scores, min_level, max_level, post_nms_top_n, name=None)

连接多级RoIs（感兴趣区域）并依据multi_scores选择N个RoIs。此操作执行以下步骤：
1、选择num_level个RoIs和scores作为输入：num_level = max_level - min_level
2、连接num_level个RoIs和scores。
3、整理scores并选择post_nms_top_n个scores。
4、通过scores中的选定指数收集RoIs。
5、通过对应的batch_id重新整理RoIs。


参数：
    - **multi_ros** (list) – 要收集的RoIs列表
    - **multi_scores** (list) - 要收集的FPN层的最低级
    - **max_level** (int) – 要收集的FPN层的最高级
    - **post_nms_top_n** (int) – 所选RoIs的数目
    - **name** (str|None) – 该层的名称（可选项）

返回：选定RoIs的输出变量

返回类型：变量(Variable)

**代码示例**

.. code-block:: python

    multi_rois = []
    multi_scores = []
    for i in range(4):
        multi_rois.append(fluid.layers.data(
            name='roi_'+str(i), shape=[4], dtype='float32', lod_level=1))
    for i in range(4):
        multi_scores.append(fluid.layers.data(
            name='score_'+str(i), shape=[1], dtype='float32', lod_level=1))
     
    fpn_rois = fluid.layers.collect_fpn_proposals(
        multi_rois=multi_rois,
        multi_scores=multi_scores,
        min_level=2,
        max_level=5,
        post_nms_top_n=2000)




.. _cn_api_fluid_layers_density_prior_box:

density_prior_box
-------------------------------

.. py:function:: paddle.fluid.layers.density_prior_box(input, image, densities=None, fixed_sizes=None, fixed_ratios=None, variance=[0.1, 0.1, 0.2, 0.2], clip=False, steps=[0.0, 0.0], offset=0.5, flatten_to_2d=False, name=None)


**Density Prior Box Operator**

为SSD算法(Single Shot MultiBox Detector)生成density prior box。
每个input的位置产生N个prior box，其中，N通过densities, fixed_sizes and fixed_ratios
的量来决定。在每个input位置附近的box center格点，通过此op生成。格点坐标由densities决定，
density prior box的量由fixed_sizes and fixed_ratios决定。显然地，fixed_sizes
和densities相等。对于densities中的densities_i：

.. math::

  N\_density\_prior\_box =sum(N\_fixed\_ratios * {densities\_i}^2)


参数：
  - **input** (Variable) - 输入变量，格式为NCHW
  - **image** (Variable) - PriorBoxOp的输入图像数据，格式为NCHW
  - **densities** (list|tuple|None) - 被生成的density prior boxes的densities，此属性应该是一个整数列表或数组。默认值为None
  - **fixed_sizes** (list|tuple|None) - 被生成的density prior boxes的固定大小，此属性应该为和 :attr:`densities` 有同样长度的列表或数组。默认值为None
  - **fixed_ratios** (list|tuple|None) - 被生成的density prior boxes的固定长度，如果该属性未被设置，同时 :attr:`densities` 和 :attr:`fix_sizes` 被设置，则 :attr:`aspect_ratios` 被用于生成 density prior boxes
  - **variance** (list|tuple) - 将被用于density prior boxes编码的方差，默认值为:[0.1, 0.1, 0.2, 0.2]
  - **clip(bool)** - 是否clip超出范围的box。默认值：False
  - **step** (list|turple) - Prior boxes在宽度和高度的步长，如果step[0] == 0.0/step[1] == 0.0, input的the density prior boxes的高度/宽度的步长将被自动计算。默认值：Default: [0., 0.]
  - **offset** (float) - Prior boxes中心补偿值，默认为：0.5
  - **flatten_to_2d** (bool) - 是否将output prior boxes和方差 ``flatten`` 至2维形状，第二个dim为4。默认值：False
  - **name(str)** - density prior box op的名字，默认值: None

返回：
  tuple: 有两个变量的数组 (boxes, variances)

  boxes: PriorBox的输出density prior boxes

    当flatten_to_2d为False时，形式为[H, W, num_priors, 4]

    当flatten_to_2d为True时，形式为[H * W * num_priors, 4]

    H是输入的高度，W是输入的宽度

    num_priors是输入中每个位置的总box count

  variances:  PriorBox的expanded variance

    当flatten_to_2d为False时，形式为[H, W, num_priors, 4]

    当flatten_to_2d为True时，形式为[H * W * num_priors, 4]

    H是输入的高度，W是输入的宽度

    num_priors是输入中每个位置的总box count

**代码示例**

.. code-block:: python
    
    input = fluid.layers.data(name="input", shape=[3,6,9])
    images = fluid.layers.data(name="images", shape=[3,9,12])
    box, var = fluid.layers.density_prior_box(
        input=input,
        image=images,
        densities=[4, 2, 1],
        fixed_sizes=[32.0, 64.0, 128.0],
        fixed_ratios=[1.],
        clip=True,
        flatten_to_2d=True)











.. _cn_api_fluid_layers_detection_map:

detection_map
-------------------------------

.. py:function:: paddle.fluid.layers.detection_map(detect_res, label, class_num, background_label=0, overlap_threshold=0.3, evaluate_difficult=True, has_state=None, input_states=None, out_states=None, ap_version='integral')

检测mAP评估算子。一般步骤如下：首先，根据检测输入和标签计算TP（true positive）和FP（false positive），然后计算mAP评估值。支持'11 point'和积分mAP算法。请从以下文章中获取更多信息：

        https://sanchom.wordpress.com/tag/average-precision/

        https://arxiv.org/abs/1512.02325

参数：
        - **detect_res** （LoDTensor）- 用具有形状[M，6]的2-D LoDTensor来表示检测。每行有6个值：[label，confidence，xmin，ymin，xmax，ymax]，M是此小批量中检测结果的总数。对于每个实例，第一维中的偏移称为LoD，偏移量为N+1，如果LoD[i+1]-LoD[i]== 0，则表示没有检测到数据。
        - **label** （LoDTensor）- 2-D LoDTensor用来带有标签的真实数据。每行有6个值：[label，xmin，ymin，xmax，ymax，is_difficult]或5个值：[label，xmin，ymin，xmax，ymax]，其中N是此小批量中真实数据的总数。对于每个实例，第一维中的偏移称为LoD，偏移量为N + 1，如果LoD [i + 1] - LoD [i] == 0，则表示没有真实数据。
        - **class_num** （int）- 类的数目。
        - **background_label** （int，defalut：0）- background标签的索引，background标签将被忽略。如果设置为-1，则将考虑所有类别。
        - **overlap_threshold** （float）- 检测输出和真实数据下限的重叠阈值。
        - **evaluate_difficult** （bool，默认为true）- 通过切换来控制是否对difficult-data进行评估。
        - **has_state** （Tensor <int>）- 是shape[1]的张量，0表示忽略输入状态，包括PosCount，TruePos，FalsePos。
        - **input_states** - 如果不是None，它包含3个元素：

            1、pos_count（Tensor）是一个shape为[Ncls，1]的张量，存储每类的输入正例的数量，Ncls是输入分类的数量。此输入用于在执行多个小批量累积计算时传递最初小批量生成的AccumPosCount。当输入（PosCount）为空时，不执行累积计算，仅计算当前小批量的结果。

            2、true_pos（LoDTensor）是一个shape为[Ntp，2]的2-D LoDTensor，存储每个类输入的正实例。此输入用于在执行多个小批量累积计算时传递最初小批量生成的AccumPosCount。

            3、false_pos（LoDTensor）是一个shape为[Nfp，2]的2-D LoDTensor，存储每个类输入的负实例。此输入用于在执行多个小批量累积计算时传递最初小批量生成的AccumPosCount。

        - **out_states** - 如果不是None，它包含3个元素：

            1、accum_pos_count（Tensor）是一个shape为[Ncls，1]的Tensor，存储每个类的实例数。它结合了输入（PosCount）和从输入中的（Detection）和（label）计算的正例数。

            2、accum_true_pos（LoDTensor）是一个shape为[Ntp'，2]的LoDTensor，存储每个类的正实例。它结合了输入（TruePos）和从输入中（Detection）和（label）计算的正实例数。 。

            3、accum_false_pos（LoDTensor）是一个shape为[Nfp'，2]的LoDTensor，存储每个类的负实例。它结合了输入（FalsePos）和从输入中（Detection）和（label）计算的负实例数。

        - **ap_version** （string，默认'integral'）- AP算法类型，'integral'或'11 point'。

返回：        具有形状[1]的（Tensor），存储mAP的检测评估结果。

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











.. _cn_api_fluid_layers_detection_output:

detection_output
-------------------------------

.. py:function:: paddle.fluid.layers.detection_output(loc, scores, prior_box, prior_box_var, background_label=0, nms_threshold=0.3, nms_top_k=400, keep_top_k=200, score_threshold=0.01, nms_eta=1.0)

Detection Output Layer for Single Shot Multibox Detector(SSD)

该操作符用于获得检测结果，执行步骤如下：

    1.根据prior box框解码输入边界框（bounding box）预测

    2.通过运用多类非极大值抑制(NMS)获得最终检测结果

请注意，该操作符不将最终输出边界框剪切至图像窗口。

参数：
    - **loc** (Variable) - 一个三维张量（Tensor），维度为[N,M,4]，代表M个bounding bboxes的预测位置。N是批尺寸，每个边界框（boungding box）有四个坐标值，布局为[xmin,ymin,xmax,ymax]
    - **scores** (Variable) - 一个三维张量（Tensor），维度为[N,M,C]，代表预测置信预测。N是批尺寸，C是类别数，M是边界框数。对每类一共M个分数，对应M个边界框
    - **prior_box** (Variable) - 一个二维张量（Tensor),维度为[M,4]，存储M个框，每个框代表[xmin,ymin,xmax,ymax]，[xmin,ymin]是anchor box的左上坐标，如果输入是图像特征图，靠近坐标系统的原点。[xmax,ymax]是anchor box的右下坐标
    - **prior_box_var** (Variable) - 一个二维张量（Tensor），维度为[M,4]，存有M变量群
    - **background_label** (float) - 背景标签索引，背景标签将会忽略。若设为-1，将考虑所有类别
    - **nms_threshold** (int) - 用于NMS的临界值（threshold）
    - **nms_top_k** (int) - 基于score_threshold过滤检测后，根据置信数维持的最大检测数
    - **keep_top_k** (int) - NMS步后，每一图像要维持的总bbox数
    - **score_threshold** (float) - 临界函数（Threshold），用来过滤带有低置信数的边界框（bounding box）。若未提供，则考虑所有框
    - **nms_eta** (float) - 适应NMS的参数

返回：
  输出一个LoDTensor，形为[No,6]。每行有6个值：[label,confidence,xmin,ymin,xmax,ymax]。No是该mini-batch的总检测数。对每个实例，第一维偏移称为LoD，偏移数为N+1，N是batch size。第i个图像有LoD[i+1]-LoD[i]检测结果。如果为0，第i个图像无检测结果。如果所有图像都没有检测结果，LoD会被设置为{1}，并且输出张量只包含一个值-1。（1.3版本后对于没有检测结果的boxes, LoD的值由之前的{0}调整为{1}）

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python
    
    import paddle.fluid as fluid
    pb = fluid.layers.data(name='prior_box', shape=[10, 4],
             append_batch_size=False, dtype='float32')
    pbv = fluid.layers.data(name='prior_box_var', shape=[10, 4],
              append_batch_size=False, dtype='float32')
    loc = fluid.layers.data(name='target_box', shape=[2, 21, 4],
              append_batch_size=False, dtype='float32')
    scores = fluid.layers.data(name='scores', shape=[2, 21, 10],
              append_batch_size=False, dtype='float32')
    nmsed_outs = fluid.layers.detection_output(scores=scores,
                           loc=loc,
                           prior_box=pb,
                           prior_box_var=pbv)






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

    fpn_rois = fluid.layers.data(
        name='data', shape=[4], dtype='float32', lod_level=1)
    multi_rois, restore_ind = fluid.layers.distribute_fpn_proposals(
        fpn_rois=fpn_rois,
        min_level=2,
        max_level=5,
        refer_level=4,
        refer_scale=224)



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





.. _cn_api_fluid_layers_generate_proposal_labels:

generate_proposal_labels
-------------------------------

.. py:function:: paddle.fluid.layers.generate_proposal_labels(rpn_rois, gt_classes, is_crowd, gt_boxes, im_info, batch_size_per_im=256, fg_fraction=0.25, fg_thresh=0.25, bg_thresh_hi=0.5, bg_thresh_lo=0.0, bbox_reg_weights=[0.1, 0.1, 0.2, 0.2], class_nums=None, use_random=True, is_cls_agnostic=False, is_cascade_rcnn=False)

**该函数可以应用于 Faster-RCNN 网络，生成建议标签。**

该函数可以根据 ``GenerateProposals`` 的输出结果，即bounding boxes（区域框），groundtruth（正确标记数据）来对foreground boxes和background boxes进行采样，并计算loss值。

RpnRois 是RPN的输出box， 并由 ``GenerateProposals`` 来进一步处理, 这些box将与groundtruth boxes合并， 并根据 ``batch_size_per_im`` 和 ``fg_fraction`` 进行采样。

如果一个实例具有大于 ``fg_thresh`` (前景重叠阀值)的正确标记重叠，那么它会被认定为一个前景样本。
如果一个实例具有的正确标记重叠大于 ``bg_thresh_lo`` 且小于 ``bg_thresh_hi`` (详见参数说明)，那么它将被认定为一个背景样本。
在所有前景、背景框（即Rois regions of interest 直译：有意义的区域）被选择后，我们接着采用随机采样的方法来确保前景框数量不多于 batch_size_per_im * fg_fraction 。

对Rois中的每个box, 我们给它分配类标签和回归目标(box label)。最后 ``bboxInsideWeights`` 和 ``BboxOutsideWeights`` 用来指明是否它将影响训练loss值。

参数:
  - **rpn_rois** (Variable) – 形为[N, 4]的二维LoDTensor。 N 为 ``GenerateProposals`` 的输出结果, 其中各元素为 :math:`[x_{min}, y_{min}, x_{max}, y_{max}]` 格式的边界框
  - **gt_classes** (Variable) – 形为[M, 1]的二维LoDTensor。 M 为正确标记数据数目, 其中各元素为正确标记数据的类别标签
  - **is_crowd** (Variable) – 形为[M, 1]的二维LoDTensor。M 为正确标记数据数目, 其中各元素为一个标志位，表明一个正确标记数据是不是crowd
  - **gt_boxes** (Variable) – 形为[M, 4]的二维LoDTensor。M 为正确标记数据数目, 其中各元素为 :math:`[x_{min}, y_{min}, x_{max}, y_{max}]` 格式的边界框
  - **im_info** (Variable) – 形为[B, 3]的二维LoDTensor。B 为输入图片的数目, 各元素由 im_height, im_width, im_scale 组成.
  - **batch_size_per_im** (int) – 每张图片的Rois batch数目
  - **fg_fraction** (float) – Foreground前景在 ``batch_size_per_im`` 中所占比例
  - **fg_thresh** (float) – 前景重叠阀值，用于选择foreground前景样本
  - **bg_thresh_hi** (float) – 背景重叠阀值的上界，用于筛选背景样本
  - **bg_thresh_lo** (float) – 背景重叠阀值的下界，用于筛选背景样本O
  - **bbox_reg_weights** (list|tuple) – Box 回归权重
  - **class_nums** (int) – 种类数目
  - **use_random** (bool) – 是否使用随机采样来选择foreground（前景）和background（背景） boxes（框）
  - **is_cls_agnostic** （bool）- 未知类别的bounding box回归，仅标识前景和背景框
  - **is_cascade_rcnn** （bool）- 是否为 cascade RCNN 模型，为True时采样策略发生变化

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    rpn_rois = fluid.layers.data(name='rpn_rois', shape=[2, 4],
                   append_batch_size=False, dtype='float32')
    gt_classes = fluid.layers.data(name='gt_classes', shape=[8, 1],
                   append_batch_size=False, dtype='float32')
    is_crowd = fluid.layers.data(name='is_crowd', shape=[8, 1],
                   append_batch_size=False, dtype='float32')
    gt_boxes = fluid.layers.data(name='gt_boxes', shape=[8, 4],
                   append_batch_size=False, dtype='float32')
    im_info = fluid.layers.data(name='im_info', shape=[10, 3],
                   append_batch_size=False, dtype='float32')
    rois, labels_int32, bbox_targets, bbox_inside_weights,
    bbox_outside_weights = fluid.layers.generate_proposal_labels(
                   rpn_rois, gt_classes, is_crowd, gt_boxes, im_info,
                   class_nums=10)











.. _cn_api_fluid_layers_generate_proposals:

generate_proposals
-------------------------------

.. py:function:: paddle.fluid.layers.generate_proposals(scores, bbox_deltas, im_info, anchors, variances, pre_nms_top_n=6000, post_nms_top_n=1000, nms_thresh=0.5, min_size=0.1, eta=1.0, name=None)

生成proposal的Faster-RCNN

该操作根据每个框为foreground（前景）对象的概率，并且通过anchors来计算这些框，进而提出RoI。Bbox_deltais和一个objects的分数作为是RPN的输出。最终 ``proposals`` 可用于训练检测网络。

为了生成 ``proposals`` ，此操作执行以下步骤：

        1、转置和调整bbox_deltas的分数和大小为（H * W * A，1）和（H * W * A，4）。

        2、计算方框位置作为 ``proposals`` 候选框。

        3、剪辑框图像。

        4、删除小面积的预测框。

        5、应用NMS以获得最终 ``proposals`` 作为输出。

参数：
        - **scores** (Variable)- 是一个shape为[N，A，H，W]的4-D张量，表示每个框成为object的概率。N是批量大小，A是anchor数，H和W是feature map的高度和宽度。
        - **bbox_deltas** （Variable）- 是一个shape为[N，4 * A，H，W]的4-D张量，表示预测框位置和anchor位置之间的差异。
        - **im_info** （Variable）- 是一个shape为[N，3]的2-D张量，表示N个批次原始图像的信息。信息包含原始图像大小和 ``feature map`` 的大小之间高度，宽度和比例。
        - **anchors** （Variable）- 是一个shape为[H，W，A，4]的4-D Tensor。H和W是 ``feature map`` 的高度和宽度，
        - **num_anchors** - 是每个位置的框的数量。每个anchor都是以非标准化格式（xmin，ymin，xmax，ymax）定义的。
        - **variances** （Variable）- anchor的方差，shape为[H，W，num_priors，4]。每个方差都是（xcenter，ycenter，w，h）这样的格式。
        - **pre_nms_top_n** （float）- 每个图在NMS之前要保留的总框数。默认为6000。
        - **post_nms_top_n** （float）- 每个图在NMS后要保留的总框数。默认为1000。
        - **nms_thresh** （float）- NMS中的阈值，默认为0.5。
        - **min_size** （float）- 删除高度或宽度小于min_size的预测框。默认为0.1。
        - **eta** （float）- 在自适应NMS中应用，如果自适应阈值> 0.5，则在每次迭代中使用adaptive_threshold = adaptive_treshold * eta。

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    scores = fluid.layers.data(name='scores', shape=[2, 4, 5, 5],
                 append_batch_size=False, dtype='float32')
    bbox_deltas = fluid.layers.data(name='bbox_deltas', shape=[2, 16, 5, 5],
                 append_batch_size=False, dtype='float32')
    im_info = fluid.layers.data(name='im_info', shape=[2, 3],
                 append_batch_size=False, dtype='float32')
    anchors = fluid.layers.data(name='anchors', shape=[5, 5, 4, 4],
                 append_batch_size=False, dtype='float32')
    variances = fluid.layers.data(name='variances', shape=[5, 5, 10, 4],
                 append_batch_size=False, dtype='float32')
    rois, roi_probs = fluid.layers.generate_proposals(scores, bbox_deltas,
                 im_info, anchors, variances)









.. _cn_api_fluid_layers_iou_similarity:

iou_similarity
-------------------------------

.. py:function:: paddle.fluid.layers.iou_similarity(x, y, name=None)

**IOU Similarity Operator**

计算两个框列表的intersection-over-union(IOU)。框列表‘X’应为LoDTensor，‘Y’是普通张量，X成批输入的所有实例共享‘Y’中的框。给定框A和框B，IOU的运算如下：

.. math::
    IOU(A, B) = \frac{area(A\cap B)}{area(A)+area(B)-area(A\cap B)}

参数：
    - **x** (Variable,默认LoDTensor,float类型) - 框列表X是二维LoDTensor，shape为[N,4],存有N个框，每个框代表[xmin,ymin,xmax,ymax],X的shape为[N,4]。如果输入是图像特征图,[xmin,ymin]市框的左上角坐标，接近坐标轴的原点。[xmax,ymax]是框的右下角坐标。张量可以包含代表一批输入的LoD信息。该批的一个实例能容纳不同的项数
    - **y** (Variable,张量，默认float类型的张量) - 框列表Y存有M个框，每个框代表[xmin,ymin,xmax,ymax],X的shape为[N,4]。如果输入是图像特征图,[xmin,ymin]市框的左上角坐标，接近坐标轴的原点。[xmax,ymax]是框的右下角坐标。张量可以包含代表一批输入的LoD信息。

返回：iou_similarity操作符的输出，shape为[N,M]的张量，代表一对iou分数

返回类型：out(Variable)

**代码示例**

..  code-block:: python

        import paddle.fluid as fluid

        x = fluid.layers.data(name='x', shape=[4], dtype='float32')
        y = fluid.layers.data(name='y', shape=[4], dtype='float32')
        iou = fluid.layers.iou_similarity(x=x, y=y)






.. _cn_api_fluid_layers_multi_box_head:

multi_box_head
-------------------------------

.. py:function:: paddle.fluid.layers.multi_box_head(inputs, image, base_size, num_classes, aspect_ratios, min_ratio=None, max_ratio=None, min_sizes=None, max_sizes=None, steps=None, step_w=None, step_h=None, offset=0.5, variance=[0.1, 0.1, 0.2, 0.2], flip=True, clip=False, kernel_size=1, pad=0, stride=1, name=None, min_max_aspect_ratios_order=False)

生成SSD（Single Shot MultiBox Detector）算法的候选框。有关此算法的详细信息，请参阅SSD论文 `SSD：Single Shot MultiBox Detector <https://arxiv.org/abs/1512.02325>`_ 的2.2节。

参数：
        - **inputs** （list | tuple）- 输入变量列表，所有变量的格式为NCHW。
        - **image** （Variable）- PriorBoxOp的输入图像数据，布局为NCHW。
        - **base_size** （int）- base_size用于根据 ``min_ratio`` 和 ``max_ratio`` 来获取 ``min_size`` 和 ``max_size`` 。
        - **num_classes** （int）- 类的数量。
        - **aspect_ratios** （list | tuple）- 生成候选框的宽高比。 ``input`` 和 ``aspect_ratios`` 的长度必须相等。
        - **min_ratio** （int）- 生成候选框的最小比率。
        - **max_ratio** （int）- 生成候选框的最大比率。
        - **min_sizes** （list | tuple | None）- 如果len（输入）<= 2，则必须设置 ``min_sizes`` ，并且 ``min_sizes`` 的长度应等于输入的长度。默认值：无。
        - **max_sizes** （list | tuple | None）- 如果len（输入）<= 2，则必须设置 ``max_sizes`` ，并且 ``min_sizes`` 的长度应等于输入的长度。默认值：无。
        - **steps** （list | tuple）- 如果step_w和step_h相同，则step_w和step_h可以被steps替换。
        - **step_w** （list | tuple）- 候选框跨越宽度。如果step_w [i] == 0.0，将自动计算输跨越入[i]宽度。默认值：无。
        - **step_h** （list | tuple）- 候选框跨越高度，如果step_h [i] == 0.0，将自动计算跨越输入[i]高度。默认值：无。
        - **offset** （float）- 候选框中心偏移。默认值：0.5
        - **variance** （list | tuple）- 在候选框编码的方差。默认值：[0.1,0.1,0.2,0.2]。
        - **flip** （bool）- 是否翻转宽高比。默认值：false。
        - **clip** （bool）- 是否剪切超出边界的框。默认值：False。
        - **kernel_size** （int）- conv2d的内核大小。默认值：1。
        - **pad** （int | list | tuple）- conv2d的填充。默认值：0。
        - **stride** （int | list | tuple）- conv2d的步长。默认值：1，
        - **name** （str）- 候选框的名称。默认值：无。
        - **min_max_aspect_ratios_order** （bool）- 如果设置为True，则输出候选框的顺序为[min，max，aspect_ratios]，这与Caffe一致。请注意，此顺序会影响卷积层后面的权重顺序，但不会影响最终检测结果。默认值：False。

返回：一个带有四个变量的元组，（mbox_loc，mbox_conf，boxes, variances）:

    - **mbox_loc** ：预测框的输入位置。布局为[N，H * W * Priors，4]。其中 ``Priors`` 是每个输位置的预测框数。

    - **mbox_conf** ：预测框对输入的置信度。布局为[N，H * W * Priors，C]。其中 ``Priors`` 是每个输入位置的预测框数，C是类的数量。

    - **boxes** ： ``PriorBox`` 的输出候选框。布局是[num_priors，4]。 ``num_priors`` 是每个输入位置的总框数。

    - **variances** ： ``PriorBox`` 的方差。布局是[num_priors，4]。 ``num_priors`` 是每个输入位置的总窗口数。

返回类型：元组（tuple）

**代码示例**

..  code-block:: python
        
        import paddle.fluid as fluid
     
        images = fluid.layers.data(name='data', shape=[3, 300, 300], dtype='float32')
        conv1 = fluid.layers.data(name='conv1', shape=[512, 19, 19], dtype='float32')
        conv2 = fluid.layers.data(name='conv2', shape=[1024, 10, 10], dtype='float32')
        conv3 = fluid.layers.data(name='conv3', shape=[512, 5, 5], dtype='float32')
        conv4 = fluid.layers.data(name='conv4', shape=[256, 3, 3], dtype='float32')
        conv5 = fluid.layers.data(name='conv5', shape=[256, 2, 2], dtype='float32')
        conv6 = fluid.layers.data(name='conv6', shape=[128, 1, 1], dtype='float32')
        
        mbox_locs, mbox_confs, box, var = fluid.layers.multi_box_head(
          inputs=[conv1, conv2, conv3, conv4, conv5, conv6],
          image=images,
          num_classes=21,
          min_ratio=20,
          max_ratio=90,
          aspect_ratios=[[2.], [2., 3.], [2., 3.], [2., 3.], [2.], [2.]],
          base_size=300,
          offset=0.5,
          flip=True,
          clip=True)




.. _cn_api_fluid_layers_multiclass_nms:

multiclass_nms
-------------------------------

.. py:function:: paddle.fluid.layers.multiclass_nms(bboxes, scores, score_threshold, nms_top_k, keep_top_k, nms_threshold=0.3, normalized=True, nms_eta=1.0, background_label=0, name=None)

**多分类NMS**

该运算用于对边界框（bounding box）和评分进行多类非极大值抑制（NMS）。

在NMS中，如果提供 ``score_threshold`` 阈值，则此算子贪婪地选择具有高于 ``score_threshold`` 的高分数的检测边界框（bounding box）的子集，然后如果nms_top_k大于-1，则选择最大的nms_top_k置信度分数。 接着，该算子基于 ``nms_threshold`` 和 ``nms_eta`` 参数，通过自适应阈值NMS移去与已经选择的框具有高IOU（intersection over union）重叠的框。

在NMS步骤后，如果keep_top_k大于-1，则每个图像最多保留keep_top_k个总bbox数。


参数：
    - **bboxes**  (Variable) – 支持两种类型的bbox（bounding box）:

      1. （Tensor）具有形[N，M，4]或[8 16 24 32]的3-D张量表示M个边界bbox的预测位置， N是批大小batch size。当边界框(bounding box)大小等于4时，每个边界框有四个坐标值，布局为[xmin，ymin，xmax，ymax]。
      2. （LoDTensor）形状为[M，C，4] M的三维张量是边界框的数量，C是种类数量

    - **scores**  (Variable) – 支持两种类型的分数：

      1. （tensor）具有形状[N，C，M]的3-D张量表示预测的置信度。 N是批量大小 batch size，C是种类数目，M是边界框bounding box的数量。对于每个类别，存在对应于M个边界框的总M个分数。请注意，M等于bboxes的第二维。
      2. （LoDTensor）具有形状[M，C]的2-D LoDTensor。 M是bbox的数量，C是种类数目。在这种情况下，输入bboxes应该是形为[M，C，4]的第二种情况。

    - **background_label**  (int) – 背景标签（类别）的索引，背景标签（类别）将被忽略。如果设置为-1，则将考虑所有类别。默认值：0
    - **score_threshold**  (float) – 过滤掉低置信度分数的边界框的阈值。如果没有提供，请考虑所有边界框。
    - **nms_top_k**  (int) – 根据通过score_threshold的过滤后而得的检测(detection)的置信度，所需要保留的最大检测数。
    - **nms_threshold**  (float) – 在NMS中使用的阈值。默认值：0.3 。
    - **nms_eta**  (float) – 在NMS中使用的阈值。默认值：1.0 。
    - **keep_top_k**  (int) – NMS步骤后每个图像要保留的总bbox数。 -1表示在NMS步骤之后保留所有bbox。
    - **normalized**  (bool) –  检测是否已经经过正则化。默认值：True 。
    - **name**  (str) – 多类nms op(此op)的名称，用于自定义op在网络中的命名。默认值：None 。

返回：形为[No，6]的2-D LoDTensor，表示检测(detections)结果。每行有6个值：[标签label，置信度confidence，xmin，ymin，xmax，ymax]。或形为[No，10]的2-D LoDTensor，用来表示检测结果。 每行有10个值：[标签label，置信度confidence，x1，y1，x2，y2，x3，y3，x4，y4]。 No是检测的总数。 如果对所有图像都没有检测到的box，则lod将设置为{1}，而Out仅包含一个值-1。 （1.3版本之后，当未检测到box时，lod从{0}更改为{1}）

返回类型：Out

**代码示例**

..  code-block:: python

    boxes = fluid.layers.data(name='bboxes', shape=[81, 4],
                              dtype='float32', lod_level=1)
    scores = fluid.layers.data(name='scores', shape=[81],
                              dtype='float32', lod_level=1)
    out = fluid.layers.multiclass_nms(bboxes=boxes,
                                      scores=scores,
                                      background_label=0,
                                      score_threshold=0.5,
                                      nms_top_k=400,
                                      nms_threshold=0.3,
                                      keep_top_k=200,
                                      normalized=False)



.. _cn_api_fluid_layers_polygon_box_transform:

polygon_box_transform
-------------------------------

.. py:function:: paddle.fluid.layers.polygon_box_transform(input, name=None)

PolygonBoxTransform 算子。

该算子用于将偏移坐标转变为真正的坐标。

输入是检测网络的最终几何输出。我们使用 2*n 个数来表示从 polygon_box 中的 n 个顶点(vertice)到像素位置的偏移。由于每个距离偏移包含两个数字 :math:`(x_i, y_i)` ，所以何输出包含 2*n 个通道。

参数：
    - **input** （Variable） - shape 为[batch_size，geometry_channels，height，width]的张量

返回：与输入 shape 相同

返回类型：output（Variable）

**代码示例**

..  code-block:: python

    import paddle.fluid as fluid
    input = fluid.layers.data(name='input', shape=[4, 10, 5, 5],
                              append_batch_size=False, dtype='float32')
    out = fluid.layers.polygon_box_transform(input)







.. _cn_api_fluid_layers_prior_box:

prior_box
-------------------------------
.. py:function:: paddle.fluid.layers.prior_box(input,image,min_sizes=None,max_sizes=None,aspect_ratios=[1.0],variance=[0.1,0.1,0.2,0.2],flip=False,clip=False,steps=[0.0,0.0],offset=0.5,name=None,min_max_aspect_ratios_order=False)

**Prior Box操作符**

为SSD(Single Shot MultiBox Detector)算法生成先验框。输入的每个位产生N个先验框，N由min_sizes,max_sizes和aspect_ratios的数目决定，先验框的尺寸在(min_size,max_size)之间，该尺寸根据aspect_ratios在序列中生成。

参数：
    - **input** (Variable)-输入变量，格式为NCHW
    - **image** (Variable)-PriorBoxOp的输入图像数据，布局为NCHW
    - **min_sizes** (list|tuple|float值)-生成的先验框的最小尺寸
    - **max_sizes** (list|tuple|None)-生成的先验框的最大尺寸。默认：None
    - **aspect_ratios** (list|tuple|float值)-生成的先验框的纵横比。默认：[1.]
    - **variance** (list|tuple)-先验框中的变量，会被解码。默认：[0.1,0.1,0.2,0.2]
    - **flip** (bool)-是否忽略纵横比。默认：False。
    - **clip** (bool)-是否修建溢界框。默认：False。
    - **step** (list|tuple)-先验框在width和height上的步长。如果step[0] == 0.0/step[1] == 0.0，则自动计算先验框在宽度和高度上的步长。默认：[0.,0.]
    - **offset** (float)-先验框中心位移。默认：0.5
    - **name** (str)-先验框操作符名称。默认：None
    - **min_max_aspect_ratios_order** (bool)-若设为True,先验框的输出以[min,max,aspect_ratios]的顺序，和Caffe保持一致。请注意，该顺序会影响后面卷基层的权重顺序，但不影响最后的检测结果。默认：False。

返回：
    含有两个变量的元组(boxes,variances)
    boxes:PriorBox的输出先验框。布局是[H,W,num_priors,4]。H是输入的高度，W是输入的宽度，num_priors是输入每位的总框数
    variances:PriorBox的扩展变量。布局上[H,W,num_priors,4]。H是输入的高度，W是输入的宽度，num_priors是输入每位的总框数

返回类型：元组

**代码示例**：

.. code-block:: python
    
    input = fluid.layers.data(name="input", shape=[3,6,9])
    images = fluid.layers.data(name="images", shape=[3,9,12])
    box, var = fluid.layers.prior_box(
        input=input,
        image=images,
        min_sizes=[100.],
        flip=True,
        clip=True)


.. _cn_api_fluid_layers_retinanet_detection_output:

retinanet_detection_output
-------------------------------

.. py:function:: paddle.fluid.layers.retinanet_detection_output(bboxes, scores, anchors, im_info, score_threshold=0.05, nms_top_k=1000, keep_top_k=100, nms_threshold=0.3, nms_eta=1.0)

**Retinanet的检测输出层**

此操作通过执行以下步骤获取检测结果：

1. 根据anchor框解码每个FPN级别的最高得分边界框预测。
2. 合并所有级别的顶级预测并对其应用多级非最大抑制（NMS）以获得最终检测。


参数：
    - **bboxes**  (List) – 来自多个FPN级别的张量列表。每个元素都是一个三维张量，形状[N，Mi，4]代表Mi边界框的预测位置。N是batch大小，Mi是第i个FPN级别的边界框数，每个边界框有四个坐标值，布局为[xmin，ymin，xmax，ymax]。
    - **scores**  (List) – 来自多个FPN级别的张量列表。每个元素都是一个三维张量，各张量形状为[N，Mi，C]，代表预测的置信度预测。 N是batch大小，C是类编号（不包括背景），Mi是第i个FPN级别的边界框数。对于每个边界框，总共有C个评分。
    - **anchors**  (List) – 具有形状[Mi，4]的2-D Tensor表示来自所有FPN级别的Mi anchor框的位置。每个边界框有四个坐标值，布局为[xmin，ymin，xmax，ymax]。
    - **im_info**  (Variable) – 形状为[N，3]的2-D LoDTensor表示图像信息。 N是batch大小，每个图像信息包括高度，宽度和缩放比例。
    - **score_threshold**  (float) – 用置信度分数剔除边界框的过滤阈值。
    - **nms_top_k**  (int) – 根据NMS之前的置信度保留每个FPN层的最大检测数。
    - **keep_top_k**  (int) – NMS步骤后每个图像要保留的总边界框数。 -1表示在NMS步骤之后保留所有边界框。
    - **nms_threshold**  (float) – NMS中使用的阈值.
    - **nms_eta**  (float) – adaptive NMS的参数.



返回：
检测输出是具有形状[No，6]的LoDTensor。 每行有六个值：[标签，置信度，xmin，ymin，xmax，ymax]。 No是此mini batch中的检测总数。 对于每个实例，第一维中的偏移称为LoD，偏移值为N + 1，N是batch大小。 第i个图像具有LoD [i + 1]  -  LoD [i]检测结果，如果为0，则第i个图像没有检测到结果。 如果所有图像都没有检测到结果，则LoD将设置为0，输出张量为空（None）。


返回类型：变量（Variable）

**代码示例**

.. code-block:: python

  import paddle.fluid as fluid

  bboxes = layers.data(name='bboxes', shape=[1, 21, 4],
      append_batch_size=False, dtype='float32')
  scores = layers.data(name='scores', shape=[1, 21, 10],
      append_batch_size=False, dtype='float32')
  anchors = layers.data(name='anchors', shape=[21, 4],
      append_batch_size=False, dtype='float32')
  im_info = layers.data(name="im_info", shape=[1, 3],
      append_batch_size=False, dtype='float32')
  nmsed_outs = fluid.layers.retinanet_detection_output(
                                          bboxes=[bboxes, bboxes],
                                          scores=[scores, scores],
                                          anchors=[anchors, anchors],
                                          im_info=im_info,
                                          score_threshold=0.05,
                                          nms_top_k=1000,
                                          keep_top_k=100,
                                          nms_threshold=0.3,
                                          nms_eta=1.)



.. _cn_api_fluid_layers_retinanet_target_assign:

retinanet_target_assign
-------------------------------

.. py:function:: paddle.fluid.layers.retinanet_target_assign(bbox_pred, cls_logits, anchor_box, anchor_var, gt_boxes, gt_labels, is_crowd, im_info, num_classes=1, positive_overlap=0.5, negative_overlap=0.4)

**Retinanet的目标分配层**

对于给定anchors和真实(ground-truth)框之间的Intersection-over-Union（IoU）重叠，该层可以为每个anchor分配分类和回归目标，同时这些目标标签用于训练Retinanet。每个anchor都分配有长度为num_classes的一个one-hot分类目标向量，以及一个4向量的框回归目标。分配规则如下：

1.在以下情况下，anchor被分配到真实框：
（i）它与真实框具有最高的IoU重叠，或者（ii）与任何真实框具有高于positive_overlap（0.5）的IoU重叠。

2.对于所有真实框，当其IoU比率低于negative_overlap（0.4）时，将anchor点分配给背景。

当为锚点分配了第i个类别的真实框时，其C向量目标中的第i项设置为1，所有其他条目设置为0.当anchor被分配支背景时，所有项都设置为0。未被分配的锚点不会影响训练目标。回归目标是与指定anchor相关联的已编码真实框。



参数：
    - **bbox_pred**  (Variable) – 具有形状[N，M，4]的3-D张量表示M个边界框(bounding box)的预测位置。 N是batch大小，每个边界框有四个坐标值，为[xmin，ymin，xmax，ymax]。
    - **cls_logits**  (Variable) – 具有形状[N，M，C]的3-D张量，表示预测的置信度。 N是batch大小，C是类别的数量（不包括背景），M是边界框的数量。
    - **anchor_box**  (Variable) – 具有形状[M，4]的2-D张量，存有M个框，每个框表示为[xmin，ymin，xmax，ymax]，[xmin，ymin]是anchor的左上顶部坐标，如果输入是图像特征图，则它们接近坐标系的原点。 [xmax，ymax]是anchor的右下坐标。
    - **anchor_var**  (Variable) – 具有形状[M，4]的2-D张量，存有anchor的扩展方差。
    - **gt_boxes**  (Variable) – 真实框是具有形状[Ng，4]的2D LoDTensor，Ng是mini batch中真实框的总数。
    - **gt_labels**  (variable) – 真实值标签是具有形状[Ng，1]的2D LoDTensor，Ng是mini batch输入真实值标签的总数。
    - **is_crowd**  (Variable) – 1-D LoDTensor，标志真实值是聚群。
    - **im_info**  (Variable) – 具有形状[N，3]的2-D LoDTensor。 N是batch大小，3分别为高度，宽度和比例。
    - **num_classes**  (int32) – 种类数量。
    - **positive_overlap**  (float) – 判定（anchor，gt框）对是一个正例的anchor和真实框之间最小重叠阀值。
    - **negative_overlap**  (float) – （锚点，gt框）对是负例时anchor和真实框之间允许的最大重叠阈值。


返回：
返回元组（predict_scores，predict_location，target_label，target_bbox，bbox_inside_weight，fg_num）。 predict_scores和predict_location是Retinanet的预测结果。target_label和target_bbox为真实值。 predict_location是形为[F，4]的2D张量，target_bbox的形状与predict_location的形状相同，F是前景anchor的数量。 predict_scores是具有形状[F + B，C]的2D张量，target_label的形状是[F + B，1]，B是背景anchor的数量，F和B取决于此算子的输入。 Bbox_inside_weight标志预测位置是否为假前景，形状为[F，4]。 Fg_num是focal loss所需的前景数（包括假前景）。


返回类型：tuple

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    bbox_pred = layers.data(name='bbox_pred', shape=[1, 100, 4],
                      append_batch_size=False, dtype='float32')
    cls_logits = layers.data(name='cls_logits', shape=[1, 100, 10],
                      append_batch_size=False, dtype='float32')
    anchor_box = layers.data(name='anchor_box', shape=[100, 4],
                      append_batch_size=False, dtype='float32')
    anchor_var = layers.data(name='anchor_var', shape=[100, 4],
                      append_batch_size=False, dtype='float32')
    gt_boxes = layers.data(name='gt_boxes', shape=[10, 4],
                      append_batch_size=False, dtype='float32')
    gt_labels = layers.data(name='gt_labels', shape=[10, 1],
                      append_batch_size=False, dtype='float32')
    is_crowd = fluid.layers.data(name='is_crowd', shape=[1],
                      append_batch_size=False, dtype='float32')
    im_info = fluid.layers.data(name='im_infoss', shape=[1, 3],
                      append_batch_size=False, dtype='float32')
    loc_pred, score_pred, loc_target, score_target, bbox_inside_weight, fg_num =
          fluid.layers.retinanet_target_assign(bbox_pred, cls_logits, anchor_box,
          anchor_var, gt_boxes, gt_labels, is_crowd, im_info, 10)










.. _cn_api_fluid_layers_roi_perspective_transform:

roi_perspective_transform
-------------------------------

.. py:function:: paddle.fluid.layers.roi_perspective_transform(input, rois, transformed_height, transformed_width, spatial_scale=1.0)

**ROI perspective transform操作符**

参数：
    - **input** (Variable) - ROI Perspective TransformOp的输入。输入张量的形式为NCHW。N是批尺寸，C是输入通道数，H是特征高度，W是特征宽度
    - **rois** (Variable) - 用来处理的ROIs，应该是shape的二维LoDTensor(num_rois,8)。给定[[x1,y1,x2,y2,x3,y3,x4,y4],...],(x1,y1)是左上角坐标，(x2,y2)是右上角坐标，(x3,y3)是右下角坐标，(x4,y4)是左下角坐标
    - **transformed_height** (integer) - 输出的高度
    - **transformed_width** (integer) – 输出的宽度
    - **spatial_scale** (float) - 空间尺度因子，用于缩放ROI坐标，默认：1.0。

返回：
 ``ROIPerspectiveTransformOp`` 的输出，它是一个4维张量，形为 (num_rois,channels,transformed_h,transformed_w)

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid

    x = fluid.layers.data(name='x', shape=[256, 28, 28], dtype='float32')
    rois = fluid.layers.data(name='rois', shape=[8], lod_level=1, dtype='float32')
    out = fluid.layers.roi_perspective_transform(x, rois, 7, 7, 1.0)







.. _cn_api_fluid_layers_rpn_target_assign:

rpn_target_assign
-------------------------------

.. py:function:: paddle.fluid.layers.rpn_target_assign(bbox_pred, cls_logits, anchor_box, anchor_var, gt_boxes, is_crowd, im_info, rpn_batch_size_per_im=256, rpn_straddle_thresh=0.0, rpn_fg_fraction=0.5, rpn_positive_overlap=0.7, rpn_negative_overlap=0.3, use_random=True)

在Faster-RCNN检测中为区域检测网络（RPN）分配目标层。

对于给定anchors和真实框之间的IoU重叠，该层可以为每个anchors做分类和回归，这些target labels用于训练RPN。classification targets是二进制的类标签（是或不是对象）。根据Faster-RCNN的论文，positive labels有两种anchors：

(i) anchor/anchors与真实框具有最高IoU重叠；

(ii) 具有IoU重叠的anchors高于带有任何真实框（ground-truth box）的rpn_positive_overlap0（0.7）。

请注意，单个真实框（ground-truth box）可以为多个anchors分配正标签。对于所有真实框（ground-truth box），非正向anchor是指其IoU比率低于rpn_negative_overlap（0.3）。既不是正也不是负的anchors对训练目标没有价值。回归目标是与positive anchors相关联而编码的图片真实框。

参数：
        - **bbox_pred** （Variable）- 是一个shape为[N，M，4]的3-D Tensor，表示M个边界框的预测位置。N是批量大小，每个边界框有四个坐标值，即[xmin，ymin，xmax，ymax]。
        - **cls_logits** （Variable）- 是一个shape为[N，M，1]的3-D Tensor，表示预测的置信度。N是批量大小，1是frontground和background的sigmoid，M是边界框的数量。
        - **anchor_box** （Variable）- 是一个shape为[M，4]的2-D Tensor，它拥有M个框，每个框可表示为[xmin，ymin，xmax，ymax]，[xmin，ymin]是anchor框的左上部坐标，如果输入是图像特征图，则它们接近坐标系的原点。 [xmax，ymax]是anchor框的右下部坐标。
        - **anchor_var** （Variable）- 是一个shape为[M，4]的2-D Tensor，它拥有anchor的expand方差。
        - **gt_boxes** （Variable）- 真实边界框是一个shape为[Ng，4]的2D LoDTensor，Ng是小批量输入的真实框（bbox）总数。
        - **is_crowd** （Variable）- 1-D LoDTensor，表示（groud-truth）是密集的。
        - **im_info** （Variable）- 是一个形为[N，3]的2-D LoDTensor。N是batch大小，第二维上的3维分别代表高度，宽度和比例(scale)
        - **rpn_batch_size_per_im** （int）- 每个图像中RPN示例总数。
        - **rpn_straddle_thresh** （float）- 通过straddle_thresh像素删除出现在图像外部的RPN anchor。
        - **rpn_fg_fraction** （float）- 为foreground（即class> 0）RoI小批量而标记的目标分数，第0类是background。
        - **rpn_positive_overlap** （float）- 对于一个正例的（anchor, gt box）对，是允许anchors和所有真实框之间最小重叠的。
        - **rpn_negative_overlap** （float）- 对于一个反例的（anchor, gt box）对，是允许anchors和所有真实框之间最大重叠的。

返回:

返回元组 (predicted_scores, predicted_location, target_label, target_bbox, bbox_inside_weight) :
   - **predicted_scores** 和 **predicted_location** 是RPN的预测结果。 **target_label** 和 **target_bbox** 分别是真实准确数据(ground-truth)。
   - **predicted_location** 是一个形为[F，4]的2D Tensor， **target_bbox** 的形与 **predicted_location** 相同，F是foreground anchors的数量。
   - **predicted_scores** 是一个shape为[F + B，1]的2D Tensor， **target_label** 的形与 **predict_scores** 的形相同，B是background anchors的数量，F和B取决于此算子的输入。
   - **Bbox_inside_weight** 标志着predicted_loction是否为fake_fg（假前景），其形为[F,4]。

返回类型：        元组(tuple)


**代码示例**

..  code-block:: python

        import paddle.fluid as fluid
        bbox_pred = fluid.layers.data(name=’bbox_pred’, shape=[100, 4],
                append_batch_size=False, dtype=’float32’)
        cls_logits = fluid.layers.data(name=’cls_logits’, shape=[100, 1],
                append_batch_size=False, dtype=’float32’)
        anchor_box = fluid.layers.data(name=’anchor_box’, shape=[20, 4],
                append_batch_size=False, dtype=’float32’)
        gt_boxes = fluid.layers.data(name=’gt_boxes’, shape=[10, 4],
                append_batch_size=False, dtype=’float32’)
        is_crowd = fluid.layers.data(name='is_crowd', shape=[1],
                    append_batch_size=False, dtype='float32')
        im_info = fluid.layers.data(name='im_infoss', shape=[1, 3],
                    append_batch_size=False, dtype='float32')
        loc_pred, score_pred, loc_target, score_target, bbox_inside_weight=
                fluid.layers.rpn_target_assign(bbox_pred, cls_logits,
                        anchor_box, anchor_var, gt_boxes, is_crowd, im_info)





.. _cn_api_fluid_layers_sigmoid_focal_loss:

sigmoid_focal_loss
-------------------------------

.. py:function:: paddle.fluid.layers.sigmoid_focal_loss(x, label, fg_num, gamma=2, alpha=0.25)

**Sigmoid Focal loss损失计算**

focal损失用于解决在one-stage探测器的训练阶段存在的前景 - 背景类不平衡问题。 此运算符计算输入张量中每个元素的sigmoid值，然后计算focal损失。

focal损失计算过程：

.. math::

  loss_j = (-label_j * alpha * {(1 - \sigma(x_j))}^{gamma} * \log(\sigma(x_j)) -
  (1 - labels_j) * (1 - alpha) * {(\sigma(x_j)}^{ gamma} * \log(1 - \sigma(x_j)))
  / fg\_num, j = 1,...,K

其中，已知：

.. math::

  \sigma(x_j) = \frac{1}{1 + \exp(-x_j)}

参数：
    - **x**  (Variable) – 具有形状[N，D]的2-D张量，其中N是batch大小，D是类的数量（不包括背景）。 此输入是由前一个运算符计算出的logits张量。
    - **label**  (Variable) – 形状为[N，1]的二维张量，是所有可能的标签。
    - **fg_num**  (Variable) – 具有形状[1]的1-D张量，是前景的数量。
    - **gamma**  (float) –  用于平衡简单和复杂实例的超参数。 默认值设置为2.0。
    - **alpha**  (float) – 用于平衡正面和负面实例的超参数。 默认值设置为0.25。


返回：  具有形状[N，D]的2-D张量，即focal损失。

返回类型： out(Variable)

**代码示例**

..  code-block:: python


    import paddle.fluid as fluid

    input = fluid.layers.data(
        name='data', shape=[10,80], append_batch_size=False, dtype='float32')
    label = fluid.layers.data(
        name='label', shape=[10,1], append_batch_size=False, dtype='int32')
    fg_num = fluid.layers.data(
        name='fg_num', shape=[1], append_batch_size=False, dtype='int32')
    loss = fluid.layers.sigmoid_focal_loss(x=input,
                                           label=label,
                                           fg_num=fg_num,
                                           gamma=2.,
                                           alpha=0.25)




.. _cn_api_fluid_layers_ssd_loss:

ssd_loss
-------------------------------

.. py:function:: paddle.fluid.layers.ssd_loss(location, confidence, gt_box, gt_label, prior_box, prior_box_var=None, background_label=0, overlap_threshold=0.5, neg_pos_ratio=3.0, neg_overlap=0.5, loc_loss_weight=1.0, conf_loss_weight=1.0, match_type='per_prediction', mining_type='max_negative', normalize=True, sample_size=None)

用于SSD的对象检测算法的多窗口损失层

该层用于计算SSD的损失，给定位置偏移预测，置信度预测，候选框和真实框标签，以及实例挖掘的类型。通过执行以下步骤，返回的损失是本地化损失（或回归损失）和置信度损失（或分类损失）的加权和：

1、通过二分匹配算法查找匹配的边界框。

        1.1、计算真实框与先验框之间的IOU相似度。

        1.2、通过二分匹配算法计算匹配的边界框。

2、计算难分样本的置信度

        2.1、根据匹配的索引获取目标标签。

        2.2、计算置信度损失。

3、应用实例挖掘来获取负示例索引并更新匹配的索引。

4、分配分类和回归目标

        4.1、根据前面的框编码bbox。

        4.2、分配回归目标。

        4.3、分配分类目标。

5、计算总体客观损失。

        5.1计算置信度损失。

        5.1计算本地化损失。

        5.3计算总体加权损失。

参数：
        - **location** （Variable）- 位置预测是具有形状[N，Np，4]的3D张量，N是批量大小，Np是每个实例的预测总数。 4是坐标值的数量，布局是[xmin，ymin，xmax，ymax]。
        - **confidence**  (Variable) - 置信度预测是具有形状[N，Np，C]，N和Np的3D张量，它们与位置相同，C是类号。
        - **gt_box** （Variable）- 真实框（bbox）是具有形状[Ng，4]的2D LoDTensor，Ng是小批量输入的真实框（bbox）的总数。
        - **gt_label** （Variable）- ground-truth标签是具有形状[Ng，1]的2D LoDTensor。
        - **prior_box** （Variable）- 候选框是具有形状[Np，4]的2D张量。
        - **prior_box_var** （Variable）- 候选框的方差是具有形状[Np，4]的2D张量。
        - **background_label** （int）- background标签的索引，默认为0。
        - **overlap_threshold** （float）- 当找到匹配的框，如果 ``match_type`` 为'per_prediction'，请使用 ``overlap_threshold`` 确定额外匹配的bbox。默认为0.5。
        - **neg_pos_ratio** （float）- 负框与正框的比率，仅在 ``mining_type`` 为'max_negative'时使用，3.0由defalut使用。
        - **neg_overlap** （float）- 不匹配预测的负重叠上限。仅当mining_type为'max_negative'时使用，默认为0.5。
        - **loc_loss_weight** （float）- 本地化丢失的权重，默认为1.0。
        - **conf_loss_weight** （float）- 置信度损失的权重，默认为1.0。
        - **match_type** （str）- 训练期间匹配方法的类型应为'bipartite'或'per_prediction'，'per_prediction'由defalut提供。
        - **mining_type** （str）- 硬示例挖掘类型应该是'hard_example'或'max_negative'，现在只支持max_negative。
        - **normalize** （bool）- 是否通过输出位置的总数将SSD丢失标准化，默认为True。
        - **sample_size** （int）- 负框的最大样本大小，仅在 ``mining_type`` 为'hard_example'时使用。

返回：        具有形状[N * Np，1]，N和Np的定位损失和置信度损失的加权和与它们在位置上的相同。

抛出异常：        ``ValueError`` - 如果 ``mining_type`` 是'hard_example'，现在只支持 ``max_negative`` 的挖掘类型。

**代码示例**

..  code-block:: python

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










.. _cn_api_fluid_layers_target_assign:

target_assign
-------------------------------

.. py:function:: paddle.fluid.layers.target_assign(input, matched_indices, negative_indices=None, mismatch_value=None, name=None)

对于给定的目标边界框（bounding box）和标签（label），该操作符对每个预测赋予分类和逻辑回归目标函数以及预测权重。权重具体表示哪个预测无需贡献训练误差。

对于每个实例，根据 ``match_indices`` 和 ``negative_indices`` 赋予输入 ``out`` 和 ``out_weight``。将定输入中每个实例的行偏移称为lod，该操作符执行分类或回归目标函数，执行步骤如下：

1.根据match_indices分配所有输入

.. code-block:: text

    If id = match_indices[i][j] > 0,

        out[i][j][0 : K] = X[lod[i] + id][j % P][0 : K]
        out_weight[i][j] = 1.

    Otherwise,

        out[j][j][0 : K] = {mismatch_value, mismatch_value, ...}
        out_weight[i][j] = 0.

2.如果提供neg_indices，根据neg_indices分配out_weight：

假设neg_indices中每个实例的行偏移称为neg_lod，该实例中第i个实例和neg_indices的每个id如下：

.. code-block:: text

    out[i][id][0 : K] = {mismatch_value, mismatch_value, ...}
    out_weight[i][id] = 1.0

参数：
    - **inputs** (Variable) - 输入为三维LoDTensor，维度为[M,P,K]
    - **matched_indices** (Variable) - 张量（Tensor），整型，输入匹配索引为二维张量（Tensor），类型为整型32位，维度为[N,P]，如果MatchIndices[i][j]为-1，在第i个实例中第j列项不匹配任何行项。
    - **negative_indices** (Variable) - 输入负例索引，可选输入，维度为[Neg,1]，类型为整型32，Neg为负例索引的总数
    - **mismatch_value** (float32) - 为未匹配的位置填充值

返回：返回一个元组（out,out_weight）。out是三维张量，维度为[N,P,K],N和P与neg_indices中的N和P一致，K和输入X中的K一致。如果match_indices[i][j]存在，out_weight是输出权重，维度为[N,P,1]。

返回类型：元组（tuple）

**代码示例**：

.. code-block:: python

        import paddle.fluid as fluid
        x = fluid.layers.data(
            name='x',
            shape=[4, 20, 4],
            dtype='float',
            lod_level=1,
            append_batch_size=False)
        matched_id = fluid.layers.data(
            name='indices',
            shape=[8, 20],
            dtype='int32',
            append_batch_size=False)
        trg, trg_weight = fluid.layers.target_assign(
            x,
            matched_id,
            mismatch_value=0)






.. _cn_api_fluid_layers_yolo_box:

yolo_box
-------------------------------

.. py:function:: paddle.fluid.layers.yolo_box(x, img_size, anchors, class_num, conf_thresh, downsample_ratio, name=None)


该运算符从YOLOv3网络的输出生成YOLO检测框。

先前网络的输出形状为[N，C，H，W]，而H和W应相同，用来指定网格大小。对每个网格点预测给定的数目的框，这个数目记为S，由anchor的数量指定。 在第二维（通道维度）中，C应该等于S *（5 + class_num），class_num是源数据集中对象类别数目（例如coco数据集中的80），此外第二个（通道）维度中还有4个框位置坐标x，y，w，h，以及anchor box的one-hot key的置信度得分。

假设4个位置坐标是 :math:`t_x` ，:math:`t_y` ，:math:`t_w` ， :math:`t_h`
，则框的预测算法为：

.. math::

    b_x &= \sigma(t_x) + c_x\\
    b_y &= \sigma(t_y) + c_y\\
    b_w &= p_w e^{t_w}\\
    b_h &= p_h e^{t_h}\\

在上面的等式中， :math:`c_x` ， :math:`c_x` 是当前网格的左上角顶点坐标。 :math:`p_w` ， :math:`p_h`  由anchors指定。

每个anchor预测框的第五通道的逻辑回归值表示每个预测框的置信度得分，并且每个anchor预测框的最后class_num通道的逻辑回归值表示分类得分。 应忽略置信度低于conf_thresh的框。另外，框最终得分是置信度得分和分类得分的乘积。


.. math::

    score_{pred} = score_{conf} * score_{class}


参数：
    - **x** （Variable） -  YoloBox算子的输入张量是一个4-D张量，形状为[N，C，H，W]。第二维（C）存储每个anchor box位置坐标，每个anchor box的置信度分数和one hot key。通常，X应该是YOLOv3网络的输出
    - **img_size** （Variable） -  YoloBox算子的图像大小张量，这是一个形状为[N，2]的二维张量。该张量保持每个输入图像的高度和宽度，用于对输出图像按输入图像比例调整输出框的大小
    - **anchors** （list | tuple） - anchor的宽度和高度，它将逐对解析
    - **class_num** （int） - 要预测的类数
    - **conf_thresh** （float） - 检测框的置信度得分阈值。置信度得分低于阈值的框应该被忽略
    - **downsample_ratio** （int） - 从网络输入到YoloBox操作输入的下采样率，因此应依次为第一个，第二个和第三个YoloBox运算设置该值为32,16,8
    - **name** （string） -  yolo box层的名称。默认None。

返回: 具有形状[N，M，4]的三维张量；框的坐标；以及具有形状[N，M，class_num]的三维张量；框的分类得分；

返回类型:   变量（Variable）

抛出异常:
    - TypeError  -  yolov_box的输入x必须是Variable
    - TypeError  -  yolo框的anchors参数必须是list或tuple
    - TypeError  -  yolo box的class_num参数必须是整数
    - TypeError  -  yolo框的conf_thresh参数必须是一个浮点数

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    x = fluid.layers.data(name='x', shape=[255, 13, 13], dtype='float32')
    anchors = [10, 13, 16, 30, 33, 23]
    loss = fluid.layers.yolo_box(x=x, img_size=608, class_num=80, anchors=anchors,
                                    conf_thresh=0.01, downsample_ratio=32)




.. _cn_api_fluid_layers_yolov3_loss:

yolov3_loss
-------------------------------

.. py:function:: paddle.fluid.layers.yolov3_loss(x, gt_box, gt_label, anchors, anchor_mask, class_num, ignore_thresh, downsample_ratio, gt_score=None, use_label_smooth=True, name=None)

该运算通过给定的预测结果和真实框生成yolov3损失。

之前的网络的输出形状为[N，C，H，W]，而H和W应该相同，用来指定网格(grid)大小。每个网格点预测给定的数目的边界框(bounding boxes)，这个给定的数字由每个尺度中 ``anchors`` 簇的个数指定，我们将它记为S。在第二维（表示通道的维度）中，C的值应为S *（class_num + 5），class_num是源数据集的对象种类数（如coco中为80），另外，除了存储4个边界框位置坐标x，y，w，h，还包括边界框以及每个anchor框的one-hot关键字的置信度得分。

假设有四个表征位置的坐标为 :math:`t_x, t_y, t_w, t_h` ,那么边界框的预测将会如下定义:

         $$
         b_x = \\sigma(t_x) + c_x
         $$
         $$
         b_y = \\sigma(t_y) + c_y
         $$
         $$
         b_w = p_w e^{t_w}
         $$
         $$
         b_h = p_h e^{t_h}
         $$

在上面的等式中， :math:`c_x, c_y` 是当前网格的左上角, :math:`p_w, p_h` 由anchors指定。
至于置信度得分，它是anchor框和真实框之间的IoU的逻辑回归值，anchor框的得分最高为1，此时该anchor框对应着最大IoU。
如果anchor框之间的IoU大于忽略阀值ignore_thresh，则该anchor框的置信度评分损失将会被忽略。
         
因此，yolov3损失包括三个主要部分，框位置损失，目标性损失，分类损失。L1损失用于
框坐标（w，h），同时，sigmoid交叉熵损失用于框坐标（x，y），目标性损失和分类损失。
         
每个真实框在所有anchor中找到最匹配的anchor，预测各anchor框都将会产生所有三种损失的计算，但是没有匹配GT box(ground truth box真实框)的anchor的预测只会产生目标性损失。

为了权衡大框(box)和小(box)之间的框坐标损失，框坐标损失将与比例权重相乘而得。即：

         $$
         weight_{box} = 2.0 - t_w * t_h
         $$

最后的loss值将如下计算:

         $$
         loss = (loss_{xy} + loss_{wh}) * weight_{box} + loss_{conf} + loss_{class}
         $$


当 ``use_label_smooth`` 设置为 ``True`` 时，在计算分类损失时将平滑分类目标，将正样本的目标平滑到1.0-1.0 / class_num，并将负样本的目标平滑到1.0 / class_num。

如果给出了 ``GTScore`` 表示真实框的mixup得分，那么真实框所产生的所有损失将乘以其混合得分。



参数：
    - **x**  (Variable) – YOLOv3损失运算的输入张量，这是一个形状为[N，C，H，W]的四维张量。H和W应该相同，第二维（C）存储框的位置信息，以及每个anchor box的置信度得分和one-hot分类
    - **gt_box**  (Variable) – 真实框，应该是[N，B，4]的形状。第三维用来承载x、y、w、h，其中 x, y是真实框的中心坐标，w, h是框的宽度和高度，且x、y、w、h将除以输入图片的尺寸，缩放到[0,1]区间内。 N是batch size，B是图像中所含有的的最多的box数目
    - **gt_label**  (Variable) – 真实框的类id，应该形为[N，B]。
    - **anchors**  (list|tuple) – 指定anchor框的宽度和高度，它们将逐对进行解析
    - **anchor_mask**  (list|tuple) – 当前YOLOv3损失计算中使用的anchor的mask索引
    - **class_num**  (int) – 要预测的类数
    - **ignore_thresh**  (float) – 一定条件下忽略某框置信度损失的忽略阈值
    - **downsample_ratio**  (int) – 从网络输入到YOLOv3 loss输入的下采样率，因此应为第一，第二和第三个YOLOv3损失运算设置32,16,8
    - **name** (string) – yolov3损失层的命名
    - **gt_score** （Variable） - 真实框的混合得分，形为[N，B]。 默认None。
    - **use_label_smooth** (bool） - 是否使用平滑标签。 默认为True


返回: 具有形状[N]的1-D张量，yolov3损失的值

返回类型:   变量（Variable）

抛出异常:
    - ``TypeError``  – yolov3_loss的输入x必须是Variable
    - ``TypeError``  – 输入yolov3_loss的gtbox必须是Variable
    - ``TypeError``  – 输入yolov3_loss的gtlabel必须是None或Variable
    - ``TypeError``  – 输入yolov3_loss的gtscore必须是Variable
    - ``TypeError``  – 输入yolov3_loss的anchors必须是list或tuple
    - ``TypeError``  – 输入yolov3_loss的class_num必须是整数integer类型
    - ``TypeError``  – 输入yolov3_loss的ignore_thresh必须是一个浮点数float类型
    - ``TypeError``  – 输入yolov3_loss的use_label_smooth必须是bool型

**代码示例**

.. code-block:: python

    x = fluid.layers.data(name='x', shape=[255, 13, 13], dtype='float32')
    gt_box = fluid.layers.data(name='gtbox', shape=[6, 4], dtype='float32')
    gt_label = fluid.layers.data(name='gtlabel', shape=[6], dtype='int32')
    gt_score = fluid.layers.data(name='gtscore', shape=[6], dtype='float32')
    anchors = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]
    anchor_mask = [0, 1, 2]
    loss = fluid.layers.yolov3_loss(x=x, gt_box=gt_box, gt_label=gt_label,
                                    gt_score=gt_score, anchors=anchors,
                                    anchor_mask=anchor_mask, class_num=80,
                                    ignore_thresh=0.7, downsample_ratio=32)








