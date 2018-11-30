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

    公式
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
