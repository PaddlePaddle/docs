.. _cn_api_fluid_layers_affine_grid:

affine_grid
-------------------------------

.. py:function:: paddle.fluid.layers.affine_grid(theta, out_shape, name=None)

该OP用于生成仿射变换前后的feature maps的坐标映射关系。在视觉应用中，根据该OP得到的映射关系，将输入feature map的像素点变换到对应的坐标，就得到了经过仿射变换的feature map。

参数：
  - **theta** (Variable) - Shape为 ``[batch_size, 2, 3]`` 的Tensor，表示batch_size个 ``2X3`` 的变换矩阵。数据类型支持float32，float64。
  - **out_shape** (Variable | list | tuple) - 类型可以是1-D Tensor、list或tuple。用于表示在仿射变换中的输出的shape，其格式 ``[N, C, H, W]`` ，分别为输出feature map的batch size、channel数量、高和宽。数据类型支持int32。
 - **name** (None|str) – 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:api_guide_Name ，默认值为None。

返回： Variable。Shape为 ``[N, H, W, 2]`` 的4-D Tensor，表示仿射变换前后的坐标的映射关系。其中，N、H、W分别为仿射变换中输出feature map的batch size、高和宽。 数据类型与 ``theta`` 一致。

返回类型：Variable

**代码示例：**

.. code-block:: python

    import paddle.fluid as fluid
    theta = fluid.layers.data(name="x", shape=[2, 3], dtype="float32")
    out_shape = fluid.layers.data(name="y", shape=[-1], dtype="float32")
    data = fluid.layers.affine_grid(theta, out_shape)
    # or
    data = fluid.layers.affine_grid(theta, [5, 3, 28, 28])
