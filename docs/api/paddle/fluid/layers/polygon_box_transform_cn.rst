.. _cn_api_fluid_layers_polygon_box_transform:

polygon_box_transform
-------------------------------

.. py:function:: paddle.fluid.layers.polygon_box_transform(input, name=None)




**PolygonBoxTransform 算子**

该op用于将偏移坐标改变为真实的坐标。

输入4-D Tensor是检测网络最终的几何输出。我们使用 2*n 个数来表示从 polygon_box 中的 n 个顶点(vertice)到像素位置的偏移。由于每个距离偏移包含两个数 :math:`(x_i, y_i)`，所以几何输出通道数为 2*n。

参数
::::::::::::

    - **input** （Variable） - 形状为 :math:`[batch\_size，geometry\_channels，height，width]` 的4-D Tensor，数据类型为float32或float64。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
polygon_box_transform输出的真实坐标，是一个 4-D Tensor。数据类型为float32或float64。

返回类型
::::::::::::
Variable

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.polygon_box_transform