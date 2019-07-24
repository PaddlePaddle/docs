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






