.. _cn_api_fluid_layers_tensor_array_to_tensor:

tensor_array_to_tensor
-------------------------------

.. py:function:: paddle.fluid.layers.tensor_array_to_tensor(input, axis=1, name=None, use_stack=False)




该OP将 ``input`` 这个LoDTensorArray中的所有Tensor沿 ``axis`` 指定的轴进行拼接（concat）或堆叠（stack）。

示例：

::
    
    - 案例 1：

        给定：
            
            input.data = {[[0.6, 0.1, 0.3],
                           [0.5, 0.3, 0.2]],
                          [[1.3],
                           [1.8]],
                          [[2.3, 2.1],
                           [2.5, 2.4]]}

            axis = 1, use_stack = False

        结果：

            output.data = [[0.6, 0.1, 0.3, 1.3, 2.3, 2.1],
                           [0.5, 0.3, 0.2, 1.8, 2.5, 2.4]]

            output_index.data = [3, 1, 2]

    - 案例 2：

        给定：
            
            input.data = {[[0.6, 0.1],
                           [0.5, 0.3]],
                          [[0.3, 1.3],
                           [0.2, 1.8]],
                          [[2.3, 2.1],
                           [2.5, 2.4]]}

            axis = 1, use_stack = False

        结果：

            output.data = [[[0.6, 0.1]
                            [0.3, 1.3]
                            [2.3, 2.1],
                           [[0.5, 0.3]
                            [0.2, 1.8]
                            [2.5, 2.4]]]

            output_index.data = [2, 2, 2]

参数
::::::::::::

  - **input** (Variable) - 输入的LoDTensorArray。支持的数据类型为：float32、float64、int32、int64。
  - **axis** (int，可选) - 指定对输入Tensor进行运算的轴，``axis`` 的有效范围是[-R, R)，R是输入 ``input`` 中Tensor的Rank，``axis`` 为负时与 ``axis`` +R 等价。默认值为1。
  - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。
  - **use_stack** (bool，可选) – 指明使用stack或concat进行运算，若为stack模式，要求LoDTensorArray中的所有Tensor具有相同的形状。默认值为False。

返回
::::::::::::
Variable的二元组，包含了两个Tensor。第一个Tensor表示对数组内的元素进行stack或concat的输出结果，数据类型与数组中的Tensor相同；第二个Tensor包含了数组中各Tensor在 `axis` 维度的大小，数据类型为int32。

返回类型
::::::::::::
 tuple

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.tensor_array_to_tensor