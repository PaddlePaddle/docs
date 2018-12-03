.. _cn_api_fluid_layers_anchor_generator:

anchor_generator
>>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.anchor_generator(input, anchor_sizes=None, aspect_ratios=None, variance=[0.1, 0.1, 0.2, 0.2], stride=None, offset=0.5, name=None)

**Anchor generator operator**

为快速RCNN算法生成锚，输入的每一位产生N个锚，N=size(anchor_sizes)*size(aspect_ratios)。生成锚的顺序首先是aspect_ratios循环，然后是anchor_sizes循环。

参数：
    - **input** (Variable) - 输入特征图，格式为NCHW
    - **anchor_sizes** (list|tuple|float) - 生成锚的锚大小
    - **in absolute pixels** 等[64.,128.,256.,512.](给定)-实例，锚大小为64意味该锚的面积等于64*2
    - **aspect_ratios** (list|tuple|float) - 生成锚的高宽比，例如[0.5,1.0,2.0]
    - **variance** (list|tuple) - 变量，在框回归delta中使用。默认：[0.1,0.1,0.2,0.2]
    - **stride** (list|tuple) - 锚在宽度和高度方向上的步长，比如[16.0,16.0]
    - **offset** (float) - 先验框的中心位移。默认：0.5
    - **name** (str) - 先验框操作符名称。默认：None

::


    **输出锚，布局[H,W,num_anchors,4]**
        H是输入的高度，W是输入的宽度，num_priors是输入每位的框数
        每个锚格式（非正式格式）为(xmin,ymin,xmax,ymax)
    
::


    **变量(Variable):锚的扩展变量**
        布局为[H,W,num_priors,4]。H是输入的高度，W是输入的宽度，num_priors是输入每位的框数，每个变量的格式为(xcenter,ycenter)。

返回类型：锚（Variable)

**代码示例**：

.. code-block:: python

    anchor, var = anchor_generator(
    input=conv1,
    anchor_sizes=[64, 128, 256, 512],
    aspect_ratios=[0.5, 1.0, 2.0],
    variance=[0.1, 0.1, 0.2, 0.2],
    stride=[16.0, 16.0],
    offset=0.5)

.. _cn_api_fluid_layers_roi_perspective_transform:

roi_perspective_transform
>>>>>>>>>>>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.roi_perspective_transform(input, rois, transformed_height, transformed_width, spatial_scale=1.0)

**ROI perspective transform操作符**

参数：
    - **input** (Variable) - ROIPerspectiveTransformOp的输入。输入张量的形式为NCHW。N是批尺寸，C是输入通道数，H是特征高度，W是特征宽度
    - **rois** (Variable) - 用来转置的ROIs(兴趣区域)，应该是shape的二维LoDTensor(num_rois,8)。给定[[x1,y1,x2,y2,x3,y3,x4,y4],...],(x1,y1)是左上角坐标，(x2,y2)是右上角坐标，(x3,y3)是右下角坐标，(x4,y4)是左下角坐标
    - **transformed_height** - 转置输出的宽度
    - **spatial_scale** (float) - todo 默认：1.0

返回：
   ``**ROIPerspectiveTransformOp的输出，带有shape的四维张量**``
    (num_rois,channels,transformed_h,transformed_w)

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    out = fluid.layers.roi_perspective_transform(input, rois, 7, 7, 1.0)

.. _cn_api_fluid_layers_iou_similarity:

iou_similarity
>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.iou_similarity(x, y, name=None)

**IOU Similarity Operator**

计算两个框列表的intersection-over-union(IOU)。框列表‘X’应为LoDTensor，‘Y’是普通张量，X成批输入的所有实例共享‘Y’中的框。给定框A和框B，IOU的运算如下：

.. math::

    IOU(A,B) = \frac{area(A \cap B )}{area(A)+area(B)-area(A \cap B)}

参数：
    - **x** (Variable,默认LoDTensor,float类型) - 框列表X是二维LoDTensor，shape为[N,4],存有N个框，每个框代表[xmin,ymin,xmax,ymax],X的shape为[N,4]。如果输入是图像特征图,[xmin,ymin]市框的左上角坐标，接近坐标轴的原点。[xmax,ymax]是框的右下角坐标。张量可以包含代表一批输入的LoD信息。该批的一个实例能容纳不同的项数
    - **y** (Variable,张量，默认float类型的张量) - 框列表Y存有M个框，每个框代表[xmin,ymin,xmax,ymax],X的shape为[N,4]。如果输入是图像特征图,[xmin,ymin]市框的左上角坐标，接近坐标轴的原点。[xmax,ymax]是框的右下角坐标。张量可以包含代表一批输入的LoD信息。

返回：iou_similarity操作符的输出，shape为[N,M]的张量，代表一对iou分数

返回类型：out(Variable)

.. _cn_api_fluid_layers_auc:

auc
>>>>>

.. py:class:: paddle.fluid.layers.auc(input, label, curve='ROC', num_thresholds=4095, topk=1, slide_steps=1)

**Area Under the Curve(AUC) Layer**

该层根据前向输出和标签计算AUC，在二分类(binary classification)估计中广泛使用。

注：如果输入标注包含一种值，只有0或1两种情况，数据类型则强制转换成布尔值。相关定义可以在 .. _这里: https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve 找到

有两种可能的曲线：

::


    1.ROC:受试者工作特征曲线
    2.PR:准确率召回率曲线

参数：
    - **input** (Variable) - 浮点二维变量，值的范围为[0,1]。每一行降序排列。输入应为topk的输出。该变量显示了每个标签的概率。
    - **label** (Variable) - 二维整型变量，表示训练数据的标注。批尺寸的高度和宽度始终为1.
    - **curve** (str) - 曲线类型，可以为‘ROC’或‘PR’。默认‘ROC’
    - **num_thresholds** (int) - 将roc曲线离散化时使用的临界值数。默认200
    - **topk** (int) - 只有预测输出的topk数才被用于auc
    - **slide_steps** - 计算批auc时，不仅用当前步也用先前步。slide_steps=1，表示用当前步；slide_steps = 3表示用当前步和前两步；slide_steps = 0，则用所有步

返回：代表当前AUC的scalar

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    # network is a binary classification model and label the ground truth
    prediction = network(image, is_infer=True)
    auc_out=fluid.layers.auc(input=prediction, label=label)

.. _cn_api_fluid_layers_dynamic_lstmp:

dynamic_lstmp
>>>>>>>>>>>>>>
.. py:class:: paddle.fluid.layers.dynamic_lstmp(input, size, proj_size, param_attr=None, bias_attr=None, use_peepholes=True, is_reverse=False, gate_activation='sigmoid', cell_activation='tanh', candidate_activation='tanh', proj_activation='tanh', dtype='float32', name=None)

动态LSTMP层(Dynamic LSTMP Layer)

LSTMP层(具有循环映射的LSTM)在LSTM层后有一个分离的映射层，从原始隐藏状态映射到较低维的状态，用来减少参数总数，减少LSTM计算复杂度，特别是输出单元相对较大的情况下。(https://research.google.com/pubs/archive/43905.pdf)

公式如下：

.. math::

        i_t & = \sigma(W_{ix}x_{t} + W_{ir}r_{t-1} + W_{ic}c_{t-1} + b_i)
        f_t & = \sigma(W_{fx}x_{t} + W_{fr}r_{t-1} + W_{fc}c_{t-1} + b_f)
        \\tilde{c_t} & = act_g(W_{cx}x_t + W_{cr}r_{t-1} + b_c)
        o_t & = \sigma(W_{ox}x_{t} + W_{or}r_{t-1} + W_{oc}c_t + b_o)
        c_t & = f_t \odot c_{t-1} + i_t \odot \\tilde{c_t}
        h_t & = o_t \odot act_h(c_t)
        r_t & = \overline{act_h}(W_{rh}h_t)


在以上公式中：
- :math: `W`:代表权重矩阵（例如 :math:`W_{xi}`是输入门道输入的权重矩阵）
- :math: `W_{ic}`, :math:`W_{fc}`, :math:`W_{oc}` peephole connections的对角权重矩阵。在我们的实现中，外面用向量代表这些对角权重矩阵
- :math: `b` :代表偏差向量（例如 :math:`b_{i}`是输入偏差向量）
- :math: `\delta` :激活函数，比如逻辑回归函数
- :math: `i,f,o` 和 :math:`c` ：分别代表输入门，遗忘门,输出门和cell激活函数向量，四者的大小和cell输出激活函数向量 :math:`h` 的四者大小相等
- :math: `h`: 隐藏状态
- :math: `r`: 隐藏状态的循环映射
- :math: `\\tilde{c_t}`:候选隐藏状态
- :math: `\odot`: 向量的元素状态生成
- :math: `act_g` 和 :math:`act_h`: cell输入和cell输出激活函数，通常使用 :math: `tanh`
- :math: `\overline{act_h}`:映射输出的激活函数，通常用 :math: `identity`或等同的 :math:`act_h`.

将use_peepholes设置为False，断开窥视孔连接（peephole connection）。在此省略公式，详情请参照论文http://www.bioinf.jku.at/publications/older/2604.pdf

注意输入 :math: `x_{t}` 中的 :math: `W_{xi}x_{t},W_{xf}x_{t},W_{xc}x_{t},W_{xo}x_{t}` 不在此操作符中。用户选择在LSTMP层之前使用全链接层。

参数：
    - **input** (Variable) - dynamic_lstmp层的输入，支持输入序列长度为变量的倍数。该变量的张量为一个矩阵，维度为（T X 4D），T为mini-batch的总时间步长，D是隐藏大小。
    - **size** (int) - 4*隐藏大小
    - **proj_size** (int) - 投影输出的大小
    - **param_attr** (ParamAttr|None) - 可学习hidden-hidden权重和投影权重的参数属性。
        - Hidden-hidden 权重 = { :math: `W_{ch},W_{ih},W_{fh},W_{oh}` }
        - hidden-hidden权重的shape（P*4D），P是投影大小，D是隐藏大小。
        - 投影（Projection）权重 = { :math: `W_{rh}` }
        - 投影权重的shape为（D*P）
        如果设为None或者ParamAttr的一个属性，dynamic_lstm将创建ParamAttr为param_attr。如果param_attr的初始函数未设置，参数则初始化为Xavier。默认:None。
    - **bias_attr** (ParamAttr|None) - 可学习bias权重的bias属性，包含输入隐藏的bias权重和窥视孔连接权重（peephole connection）,前提是use_peepholes设为True。
        1.use_peepholes = False

        ::

            - Biases = { :math: `b_{c},b_{i},b_{f},b_{o}`}.
            - 维度为（1*4D）

        2.use_peepholes = True

        ::

            - Biases = { :math: `b_{c},b_{i},b_{f},b_{o},W_{ic},W_{fc},W_{oc}`}
            - 维度为（1*7D）
        
        如果设置为None或者ParamAttr的一个属性，dynamic_lstm将创建ParamAttr为bias_attr。bias_attr的初始函数未设置，bias则初始化为0.默认：None。
        
    - **use_peepholes** (bool) - 是否开启诊断/窥视孔链接，默认为True。
    - **is_reverse** (bool) - 是否计算反向LSTM，默认为False。
    - **gate_activation** (bool) - 输入门（input gate）、遗忘门（forget gate）和输出门（output gate）的激活函数。Choices = [“sigmoid”，“tanh”，“relu”，“identity”]，默认“sigmoid”。
    - **cell_activation** (str) - cell输出的激活函数。Choices = [“sigmoid”，“tanh”，“relu”，“identity”]，默认“tanh”。
    - **candidate_activation** (str) - 候选隐藏状态（candidate hidden state）的激活状态。Choices = [“sigmoid”，“tanh”，“relu”，“identity”]，默认“tanh”。
    - **proj_activation** (str) - 投影输出的激活函数。Choices = [“sigmoid”，“tanh”，“relu”，“identity”]，默认“tanh”。
    - **dtype** (str) - 数据类型。Choices = [“float32”，“float64”]，默认“float32”。
    - **name** (str|None) - 该层名称（可选）。若设为None，则自动为该层命名。 

返回：含有两个输出变量的元组，隐藏状态（hidden state）的投影和LSTMP的cell状态。投影的shape为（T*P），cell state的shape为（T*D），两者的LoD和输入相同。

返回类型：元组(tuple)

**代码示例**：

.. code-block:: python

    dict_dim, emb_dim = 128, 64
    data = fluid.layers.data(name='sequence', shape=[1],
                         dtype='int32', lod_level=1)
    emb = fluid.layers.embedding(input=data, size=[dict_dim, emb_dim])
    hidden_dim, proj_dim = 512, 256
    fc_out = fluid.layers.fc(input=emb, size=hidden_dim * 4,
                         act=None, bias_attr=None)
    proj_out, _ = fluid.layers.dynamic_lstmp(input=fc_out,
                                         size=hidden_dim * 4,
                                         proj_size=proj_dim,
                                         use_peepholes=False,
                                         is_reverse=True,
                                         cell_activation="tanh",
                                         proj_activation="tanh")

.. _cn_api_fluid_layers_linear_chain_crf:

linear_chain_crf
>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.linear_chain_crf(input, label, param_attr=None)

线性链条件随机场（Linear Chain CRF）

条件随机场定义间接概率图，节点代表随机变量，边代表两个变量之间的依赖。CRF学习条件概率 :math: `P\left ( Y|X \right )` ， :math: `X = \left ( x_{1},x_{2},...,x_{n} \right )` 是结构性输入，:math: `Y = \left ( y_{1},y_{2},...,y_{n} \right )` 为输入标签。

线性链条件随机场（Linear Chain CRF)是特殊的条件随机场（CRF），有利于序列标注任务。序列标注任务不为输入设定许多条件依赖。唯一的限制是输入和输出必须是现行序列。因此类似CRF的图是一个简单的链或者线，也就是线性链随机场（linear chain CRF）。

该操作符实现了线性链条件随机场（linear chain CRF）的前后向算法。详情请参照http://www.cs.columbia.edu/~mcollins/fb.pdf和http://cseweb.ucsd.edu/~elkan/250Bwinter2012/loglinearCRFs.pdf。

.. _cn_api_fluid_layers_chunk_eval:

chunk_eval
>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.chunk_eval(input, label, chunk_scheme, num_chunk_types, excluded_chunk_types=None)

块估计（Chunk Evaluator）

该功能计算并输出块检测（chunk detection）的准确率、召回率和F1值。

chunking的一些基础请参考 .. _Chunking with Support Vector Machines: https://aclanthology.info/pdf/N/N01/N01-1025.pdf

ChunkEvalOp计算块检测（chunk detection）的准确率、召回率和F1值，并支持IOB，IOE，IOBES和IO标注方案。以下是这些标注方案的命名实体（NER）标注例子：

::


    ====== ====== ======  =====  ==  ============   =====  ===== =====  ==  =========
        Li     Ming    works  at  Agricultural   Bank   of    China  in  Beijing.
    ====== ====== ======  =====  ==  ============   =====  ===== =====  ==  =========
    IO     I-PER  I-PER   O      O   I-ORG          I-ORG  I-ORG I-ORG  O   I-LOC
    IOB    B-PER  I-PER   O      O   B-ORG          I-ORG  I-ORG I-ORG  O   B-LOC
    IOE    I-PER  E-PER   O      O   I-ORG          I-ORG  I-ORG E-ORG  O   E-LOC
    IOBES  B-PER  E-PER   O      O   I-ORG          I-ORG  I-ORG E-ORG  O   S-LOC
    ====== ====== ======  =====  ==  ============   =====  ===== =====  == 

有三种块类别（命名实体类型），包括PER（人名），ORG（机构名）和LOC（地名），标签形式为标注类型（tag type）-块类型（chunk type）。

由于计算实际上用的是标签id而不是标签，需要额外注意将标签映射到相应的id，这样CheckEvalOp才可运行。关键在于id必须在列出的等式中有效。

::


    tag_type = label % num_tag_type
    chunk_type = label / num_tag_type

num_tag_type是标注规则中的标签类型数，num_chunk_type是块类型数，tag_type从下面的表格中获取值。

::


    Scheme Begin Inside End   Single
    plain   0     -      -     -
    IOB     0     1      -     -
    IOE     -     0      1     -
    IOBES   0     1      2     3

仍以NER为例，假设标注规则是IOB块类型为ORG，PER和LOC。为了满足以上等式，标签图如下：

::


    B-ORG  0
    I-ORG  1
    B-PER  2
    I-PER  3
    B-LOC  4
    I-LOC  5
    O      6

不难证明等式的块类型数为3，IOB规则中的标签类型数为2.例如I-LOC的标签id为5，I-LOC的标签类型id为1，I-LOC的块类型id为2，与等式的结果一致。

参数：
    - **input** (Variable) - 网络的输出预测
    - **label** (Variable) - 测试数据集的标签
    - **chunk_scheme** (str) - 标注规则，表示如何解码块。必须数IOB，IOE，IOBES或者plain。详情见描述
    - **num_chunk_types** (int) - 块类型数。详情见描述
    - **excluded_chunk_types** (list) - 列表包含块类型id，表示不在计数内的块类型。详情见描述

返回：元组（tuple），包含precision, recall, f1_score, num_infer_chunks, num_label_chunks, num_correct_chunks

返回类型：tuple（元组）

**代码示例**：
.. code-block:: python:

    crf = fluid.layers.linear_chain_crf(
        input=hidden, label=label, param_attr=ParamAttr(name="crfw"))
    crf_decode = fluid.layers.crf_decoding(
        input=hidden, param_attr=ParamAttr(name="crfw"))
    fluid.layers.chunk_eval(
        input=crf_decode,
        label=label,
        chunk_scheme="IOB",
        num_chunk_types=(label_dict_len - 1) / 2)

.. _cn_api_fluid_layers_conv2d:

conv2d
>>>>>>>

.. py:class:: paddle.fluid.layers.conv2d(input, num_filters, filter_size, stride=1, padding=0, dilation=1, groups=None, param_attr=None, bias_attr=None, use_cudnn=True, act=None, name=None)

卷积二维层（convolution2D layer）根据输入、滤波器（filter）、步长（stride）、填充（padding）、dilations、一组参数计算输出。输入和输出是NCHW格式，N是批尺寸，C是通道数，H是特征高度，W是特征宽度。滤波器是MCHW格式，M是输出图像通道数，C是输入图像通道数，H是滤波器高度，W是滤波器宽度。如果组数大于1，C等于输入图像通道数除以组数的结果。详情请参考UFLDL's : _卷积: http://ufldl.stanford.edu/tutorial/supervised/FeatureExtractionUsingConvolution/ 。如果提供了bias属性和激活函数类型，bias会添加到卷积（convolution）的结果中相应的激活函数会作用在最终结果上。

对每个输入X，有等式：

.. math::


    Out = \sigma \left ( W * X + b \right )

其中：

    - **X** ：输入值，NCHW格式的张量（Tensor）
    - **W** ：滤波器值，MCHW格式的张量（Tensor）
    - * ： 卷积操作
    - **b** ：Bias值，二维张量（Tensor），shape为[M,1]
    - ** :math: `\sigma` ** ：激活函数
    - ** :math: `Out` ** ：输出值，*Out*和**X**的shape可能不同

**示例**

- 输入：
    输入shape：:math: `\left ( N,C_{in},H_{in},W_{in} \right )`
    滤波器shape： :math: `\left ( C_{out},C_{in},H_{f},W_{f} \right )`
- 输出：
    输出shape： :math: `\left ( N,C_{out},H_{out},W_{out} \right )`

其中

.. math::


    H_{out} = \frac{\left ( H_{in}+2*paddings[0]-\left ( dilations[0]*\left ( H_{f}-1 \right )+1 \right ) \right )}{strides[0]}+1

    W_{out} = \frac{\left ( H_{in}+2*paddings[1]-\left ( dilations[1]*\left ( W_{f}-1 \right )+1 \right ) \right )}{strides[1]}+1

参数：
    - **input** (Variable) - 格式为[N,C,H,W]格式的输入图像
    - **num_fliters** (int) - 滤波器数。和输出图像通道相同
    - **filter_size** (int|tuple|None) - 滤波器大小。如果filter_size是一个元组，则必须包含两个整型数，（filter_size，filter_size_W）。否则，滤波器为square
    - **stride** (int|tuple) - 步长(stride)大小。如果步长（stride）为元组，则必须包含两个整型数，（stride_H,stride_W）。否则，stride_H = stride_W = stride。默认：stride = 1
    - **padding** (int|tuple) - 填充（padding）大小。如果填充（padding）为元组，则必须包含两个整型数，（padding_H,padding_W)。否则，padding_H = padding_W = padding。默认：padding = 0
    - **dilation** (int|tuple) - 膨胀（dilation）大小。如果膨胀（dialation）为元组，则必须包含两个整型数，（dilation_H,dilation_W）。否则，dilation_H = dilation_W = dilation。默认：dilation = 1
    - **groups** (int) - 卷积二维层（Conv2D Layer）的组数。根据Alex Krizhevsky的深度卷积神经网络（CNN）论文中的成组卷积：当group=2，滤波器的前一半仅和输入通道的前一半连接。滤波器的后一半仅和输入通道的后一半连接。默认：groups = 1
    - **param_attr** (ParamAttr|None) - conv2d的可学习参数/权重的参数属性。如果设为None或者ParamAttr的一个属性，conv2d创建ParamAttr为param_attr。如果param_attr的初始化函数未设置，参数则初始化为 :math: `Normal(0.0,std)`，并且std为 :math: `\left ( \frac{2.0}{filter\_elem\_num} \right )^{0.5}`。默认为None
    - **bias_attr** (ParamAttr|bool|None) - conv2d bias的参数属性。如果设为False，则没有bias加到输出。如果设为None或者ParamAttr的一个属性，conv2d创建ParamAttr为bias_attr。如果bias_attr的初始化函数未设置，bias初始化为0.默认为None
    - **use_cudnn** （bool） - 是否用cudnn核，仅当下载cudnn库才有效。默认：True
    - **act** (str) - 激活函数类型，如果设为None，则未添加激活函数。默认：None
    - **name** (str|None) - 该层名称（可选）。若设为None，则自动为该层命名。

返回：张量，存储卷积和非线性激活结果

返回类型：变量（Variable）

抛出异常：`ValueError` - 如果输入shape和filter_size，stride,padding和group不匹配。

**代码示例**：

.. code-block:: python

    data = fluid.layers.data(name='data', shape=[3, 32, 32], dtype='float32')
    conv2d = fluid.layers.conv2d(input=data, num_filters=2, filter_size=3, act="relu")

.. _cn_api_fluid_layers_conv3d:

conv3d
>>>>>>>

.. py:class:: paddle.fluid.layers.conv3d(input, num_filters, filter_size, stride=1, padding=0, dilation=1, groups=None, param_attr=None, bias_attr=None, use_cudnn=True, act=None, name=None)

卷积三维层（convolution3D layer）根据输入、滤波器（filter）、步长（stride）、填充（padding）、膨胀（dilations）、组数参数计算得到输出。输入和输出是NCHW格式，N是批尺寸，C是通道数，H是特征高度，W是特征宽度。卷积三维（Convlution3D）和卷积二维（Convlution2D）相似，但多了一维深度（depth）。如果提供了bias属性和激活函数类型，bias会添加到卷积（convolution）的结果中相应的激活函数会作用在最终结果上。

对每个输入X，有等式：

.. math::


    Out = \sigma \left ( W * X + b \right )

其中：

    - **X** ：输入值，NCHW格式的张量（Tensor）
    - **W** ：滤波器值，MCHW格式的张量（Tensor）
    - * ： 卷积操作
    - **b** ：Bias值，二维张量（Tensor），shape为[M,1]
    - ** :math: `\sigma` ** ：激活函数
    - ** :math: `Out` ** ：输出值，*Out*和**X**的shape可能不同

**示例**

- 输入：
    输入shape：:math: `\left ( N,C_{in},H_{in},W_{in} \right )`

    滤波器shape： :math: `\left ( C_{out},C_{in},H_{f},W_{f} \right )`
- 输出：
    输出shape： :math: `\left ( N,C_{out},H_{out},W_{out} \right )`

其中

.. math::


    D_{out} = \frac{\left ( D_{in}+2*paddings[0]-\left ( dilations[0]*\left ( D_{f}-1 \right )+1 \right ) \right )}{strides[0]}+1

    H_{out} = \frac{\left ( H_{in}+2*paddings[1]-\left ( dilations[1]*\left ( H_{f}-1 \right )+1 \right ) \right )}{strides[1]}+1

    W_{out} = \frac{\left ( W_{in}+2*paddings[2]-\left ( dilations[2]*\left ( W_{f}-1 \right )+1 \right ) \right )}{strides[2]}+1

参数：
    - **input** (Variable) - 格式为[N,C,H,W]格式的输入图像
    - **num_fliters** (int) - 滤波器数。和输出图像通道相同
    - **filter_size** (int|tuple|None) - 滤波器大小。如果filter_size是一个元组，则必须包含两个整型数，（filter_size，filter_size_W）。否则，滤波器为square
    - **stride** (int|tuple) - 步长(stride)大小。如果步长（stride）为元组，则必须包含两个整型数，（stride_H,stride_W）。否则，stride_H = stride_W = stride。默认：stride = 1
    - **padding** (int|tuple) - 填充（padding）大小。如果填充（padding）为元组，则必须包含两个整型数，（padding_H,padding_W)。否则，padding_H = padding_W = padding。默认：padding = 0
    - **dilation** (int|tuple) - 膨胀（dilation）大小。如果膨胀（dialation）为元组，则必须包含两个整型数，（dilation_H,dilation_W）。否则，dilation_H = dilation_W = dilation。默认：dilation = 1
    - **groups** (int) - 卷积二维层（Conv2D Layer）的组数。根据Alex Krizhevsky的深度卷积神经网络（CNN）论文中的成组卷积：当group=2，滤波器的前一半仅和输入通道的前一半连接。滤波器的后一半仅和输入通道的后一半连接。默认：groups = 1
    - **param_attr** (ParamAttr|None) - conv2d的可学习参数/权重的参数属性。如果设为None或者ParamAttr的一个属性，conv2d创建ParamAttr为param_attr。如果param_attr的初始化函数未设置，参数则初始化为 :math: `Normal(0.0,std)`，并且std为 :math: `\left ( \frac{2.0}{filter\_elem\_num} \right )^{0.5}`。默认为None
    - **bias_attr** (ParamAttr|bool|None) - conv2d bias的参数属性。如果设为False，则没有bias加到输出。如果设为None或者ParamAttr的一个属性，conv2d创建ParamAttr为bias_attr。如果bias_attr的初始化函数未设置，bias初始化为0.默认为None
    - **use_cudnn** （bool） - 是否用cudnn核，仅当下载cudnn库才有效。默认：True
    - **act** (str) - 激活函数类型，如果设为None，则未添加激活函数。默认：None
    - **name** (str|None) - 该层名称（可选）。若设为None，则自动为该层命名。

返回：张量，存储卷积和非线性激活结果

返回类型：变量（Variable）

抛出异常：`ValueError` - 如果输入shape和filter_size，stride,padding和group不匹配。

**代码示例**：

.. code-block:: python

    data = fluid.layers.data(name='data', shape=[3, 12, 32, 32], dtype='float32')
    conv3d = fluid.layers.conv3d(input=data, num_filters=2, filter_size=3, act="relu")

.. _cn_api_fluid_layers_sequence_softmax:

sequence_softmax
>>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.sequence_softmax(input, use_cudnn=False, name=None)

该函数计算每一个序列所有时间步中的softmax激活函数。每个时间步的维度应为1.输入张量的shape可为[N，1]或者[N],N是所有序列长度之和。

对mini-batch的第i序列：

.. math::

    Out\left ( X[lod[i]:lod[i+1]],: \right ) = \frac{exp(X[lod[i]:lod[i+1],:])}{\sum (exp(X[lod[i]:lod[i+1],:]))}

例如，对有3个序列（带变量长度）的mini-batch，每个包含2，3，2时间步，其lod为[0,2,5,7]，则在 :math: `X[0:2,:],X[2:5,:],X[5:7,:]`，并且N的结果为7.

参数：
    - **input** (Variable) - 输入变量，为LoDTensor
    - **use_cudnn** (bool) - 是否用cudnn核，仅当下载cudnn库才有效。默认：False
    - **name** (str|None) - 该层名称（可选）。若设为None，则自动为该层命名。默认：None

返回：sequence_softmax的输出

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    x = fluid.layers.data(name='x', shape=[7, 1],
                 dtype='float32', lod_level=1)
    x_sequence_softmax = fluid.layers.sequence_softmax(input=x)

.. _cn_api_fluid_layers_pool2d:

pool2d
>>>>>>>

.. py:class:: paddle.fluid.layers.pool2d(input, pool_size=-1, pool_type='max', pool_stride=1, pool_padding=0, global_pooling=False, use_cudnn=True, ceil_mode=False, name=None)

pooling2d操作符根据input，pooling_type和ksize，步长（stride），填充（padding）这些参数得到输出。输入X和输出Out是NCHW格式，N为批尺寸，C是通道数，H是特征高度，W是特征宽度。参数（ksize,strides,paddings）是两个元素。这两个元素分别代表高度和宽度。输入X的大小和输出Out的大小可能不一致。

例如：

输入：
    X shape：:math: `\left ( N,C_{in},H_{in},W_{in} \right )`

输出：
    Out shape：:math: `\left ( N,C_{out},H_{out},W_{out} \right )`

cell_mode = false：

.. math::

    H_{out}=\frac{(H_{in}-ksize[0]+2*paddings[0]+strides[0]-1)}{strides[0]}+1

    W_{out}=\frac{(W_{in}-ksize[1]+2*paddings[1]+strides[1]-1)}{strides[1]}+1

参数：
    - **input** (Variable) - 池化操作的输入张量。输入张量格式为NCHW，N为批尺寸，C是通道数，H是特征高度，W是特征宽度
    - **pool_size** (int) - 池化窗口的边长。所有池化窗口为正方形，边长为pool_size
    - **pool_type** (string) - 池化类型，可以是“max”对应max-pooling，“avg”对应average-pooling
    - **pool_stride** (int) - 池化层的步长
    - **pool_padding** (int) - 填充大小
    - **global_pooling** （bool，默认false）- 是否用全局池化。如果global_pooling = true，ksize和填充（padding）则被忽略
    - **use_cudnn** （bool，默认false）- 只在cudnn核中用，需要下载cudnn
    - **ceil_mode** （bool，默认false）- 是否用ceil函数计算输出高度和宽度。默认False。如果设为False，则使用floor函数
    - **name** （str|None） - 该层名称（可选）。若设为None，则自动为该层命名。

返回：池化结果

返回类型：变量（Variable）

抛出异常：
    - `ValueError` - 如果‘pool_type’既不是“max”也不是“avg”
    - `ValueError` - 如果‘global_pooling’为False并且‘pool_size’为-1
    - `ValueError` - 如果‘use_cudnn’不是bool值

**代码示例**：

.. code-block:: python

    data = fluid.layers.data(
        name='data', shape=[3, 32, 32], dtype='float32')
    conv2d = fluid.layers.pool2d(
                  input=data,
                  pool_size=2,
                  pool_type='max',
                  pool_stride=1,
                  global_pooling=False)

.. _cn_api_fluid_layers_batch_norm:

batch_norm
>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.batch_norm(input, act=None, is_test=False, momentum=0.9, epsilon=1e-05, param_attr=None, bias_attr=None, data_layout='NCHW', in_place=False, name=None, moving_mean_name=None, moving_variance_name=None, do_model_average_for_mean_and_var=False, fuse_with_relu=False)

批正则化层（Batch Normalization Layer）

可用作conv2d和全链接操作的正则化函数。该层需要的数据格式如下：

    1.NHWC[batch,in_height,in_width,in_channels]
    2.NCHW[batch,in_channels,in_height,in_width]

更多详情请参考 : _ Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift : https://arxiv.org/pdf/1502.03167.pdf

`input`是mini-batch的输入特征。

参数：
    - **input** (Variable) - 输入变量，为LoDTensor
    - **act** （string，默认None）- 激活函数类型，linear|relu|prelu|...
    - **is_test** （bool,默认False） - 用于训练
    - **momentum** （float，默认0.9）
    - **epsilon** （float，默认1e-05）
    - **param_attr** （ParamAttr|None） - batch_norm参数范围的属性，如果设为None或者是ParamAttr的一个属性，batch_norm创建ParamAttr为param_attr。如果没有设置param_attr的初始化函数，参数初始化为Xavier。默认：None
    - **bias_attr** （ParamAttr|None） - batch_norm bias参数的属性，如果设为None或者是ParamAttr的一个属性，batch_norm创建ParamAttr为bias_attr。如果没有设置bias_attr的初始化函数，参数初始化为0。默认：None
    - **data_layout** （string,默认NCHW) - NCHW |NHWC
    - **in_place** （bool，默认False）- 得出batch norm可复用记忆的输入和输出
    - **name** （string，默认None）- 该层名称（可选）。若设为None，则自动为该层命名
    - **moving_mean_name** （string，默认None）- moving_mean的名称，存储全局Mean
    - **moving_variance_name** （string，默认None）- moving_variance的名称，存储全局变量
    - **do_model_average_for_mean_and_var** （bool，默认False）- 是否为mean和variance做模型均值
    - **fuse_with_relo** （bool）- 如果为True，batch norm后该操作符执行relu

返回： 张量，在输入中运用批正则后的结果

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    hidden1 = fluid.layers.fc(input=x, size=200, param_attr='fc1.w')
    hidden2 = fluid.layers.batch_norm(input=hidden1)
