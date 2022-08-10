.. _cn_api_fluid_layers_affine_grid:

affine_grid
-------------------------------

.. py:function:: paddle.fluid.layers.affine_grid(theta, out_shape, name=None)




用于生成仿射变换前后的 feature maps 的坐标映射关系。在视觉应用中， 根据得到的映射关系，将输入 feature map 的像素点变换到对应的坐标，就得到了经过仿射变换的 feature map。

参数
::::::::::::

  - **theta** (Variable) - Shape 为 ``[batch_size, 2, 3]`` 或 ``[batch_size, 3, 4]`` 的 Tensor，表示 batch_size 个 ``2X3`` 或 ``3X4`` 的变换矩阵。数据类型支持 float32，float64。
  - **out_shape** (Variable | list | tuple) - 类型可以是 1-D Tensor、list 或 tuple。用于表示在仿射变换中的输出的 shape，其格式为 ``[N, C, H, W]`` 或 ``[N, C, D, H, W]`` ，格式 ``[N, C, H, W]`` 分别表示输出 feature map 的 batch size、channel 数量、高和宽，格式 ``[N, C, D, H, W]`` 分别表示输出 feature map 的 batch size、channel 数量、深度、高和宽。数据类型支持 int32。
 - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
 Variable。Shape 为 ``[N, H, W, 2]`` 的 4-D Tensor 或``[N, D, H, W, 3]``的 5-D Tensor，表示仿射变换前后的坐标的映射关系。输出为 4-D Tensor 时，N、H、W 分别为仿射变换中输出 feature map 的 batch size、高和宽，输出为 5D Tensor 时，N、D、H、W 分别为仿射变换中输出 feature map 的 batch size、深度、高和宽。数据类型与 ``theta`` 一致。

返回类型
::::::::::::
Variable

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.affine_grid
