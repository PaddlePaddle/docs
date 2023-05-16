.. _cn_api_fluid_layers_fill_constant:

fill_constant
-------------------------------

.. py:function:: paddle.fluid.layers.fill_constant(shape,dtype,value,force_cpu=False,out=None)




该 OP 创建一个形状为 shape 并且数据类型为 dtype 的 Tensor，同时用 ``value`` 中提供的常量初始化该 Tensor。

创建的 Tensor 的 stop_gradient 属性默认为 True。

参数
::::::::::::

    - **shape** (tuple|list|Variable)- 要创建的 LoDTensor 或者 SelectedRows 的形状。数据类型为 int32 或 int64。如果 shape 是一个列表或元组，则其元素应该是整数或形状为[]的 0-D Tensor。如果 shape 是 Variable，则它应该是一维 Tensor。
    - **dtype** (np.dtype|core.VarDesc.VarType|str)- 创建 LoDTensor 或者 SelectedRows 的数据类型，支持数据类型为 float16， float32， float64， int32， int64。
    - **value** (float|int)- 用于初始化输出 LoDTensor 或者 SelectedRows 的常量数据的值。
    - **force_cpu** (bool)- 用于标志 LoDTensor 或者 SelectedRows 是否创建在 CPU 上，默认值为 False，若设为 true，则数据必须在 CPU 上。
    - **out** (Variable，可选)- 用于存储创建的 LoDTensor 或者 SelectedRows，可以是程序中已经创建的任何 Variable。默认值为 None，此时将创建新的 Variable 来保存输出结果。


返回
::::::::::::
 根据 shape 和 dtype 创建的 Tensor。

返回类型
::::::::::::
变量（Variable）

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.fill_constant
