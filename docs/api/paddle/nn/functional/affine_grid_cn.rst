.. _cn_api_nn_functional_affine_grid:

affine_grid
-------------------------------

.. py:function:: paddle.nn.functional.affine_grid(theta, out_shape, align_corners=True, name=None)


该OP用于生成仿射变换前后的feature maps的坐标映射关系。在视觉应用中，根据该OP得到的映射关系，将输入feature map的像素点变换到对应的坐标，就得到了经过仿射变换的feature map。

参数
::::::::::::

  - **theta** (Tensor) - Shape为 ``[batch_size, 2, 3]`` 的Tensor，表示batch_size个 ``2X3`` 的变换矩阵。数据类型支持float32，float64。
  - **out_shape** (Tensor | list | tuple) - 类型可以是1-D Tensor、list或tuple。用于表示在仿射变换中的输出的shape，其格式 ``[N, C, H, W]``，分别为输出feature map的batch size、channel数量、高和宽。数据类型支持int32。
  - **align_corners** (bool, optional)：一个可选的bool型参数，如果为True，则将输入和输出张量的4个角落像素的中心对齐，并保留角点像素的值。默认值：True。
  - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
 Tensor。Shape为 ``[N, H, W, 2]`` 的4-D Tensor，表示仿射变换前后的坐标的映射关系。其中，N、H、W分别为仿射变换中输出feature map的batch size、高和宽。数据类型与 ``theta`` 一致。


代码示例
::::::::::::

COPY-FROM: paddle.nn.functional.affine_grid