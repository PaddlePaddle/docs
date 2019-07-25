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

    import paddle.fluid as fluid
    conv1 = fluid.layers.data(name='conv1', shape=[48, 16, 16], dtype='float32')
    anchor, var = fluid.layers.anchor_generator(
    input=conv1,
    anchor_sizes=[64, 128, 256, 512],
    aspect_ratios=[0.5, 1.0, 2.0],
    variance=[0.1, 0.1, 0.2, 0.2],
    stride=[16.0, 16.0],
    offset=0.5)









