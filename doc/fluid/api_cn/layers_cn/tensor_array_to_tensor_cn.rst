.. _cn_api_fluid_layers_tensor_array_to_tensor:

tensor_array_to_tensor
-------------------------------

.. py:function:: paddle.fluid.layers.tensor_array_to_tensor(input, axis=1, name=None)

此函数在指定轴上连接LodTensorArray中的元素，并将其作为输出返回。


简单示例如下：

.. code-block:: text

    Given:
    input.data = {[[0.6, 0.1, 0.3],
                   [0.5, 0.3, 0.2]],
                  [[1.3],
                   [1.8]],
                  [[2.3, 2.1],
                   [2.5, 2.4]]}

    axis = 1

    Then:
    output.data = [[0.6, 0.1, 0.3, 1.3, 2.3, 2.1],
                   [0.5, 0.3, 0.2, 1.8, 2.5, 2.4]]
    output_index.data = [3, 1, 2]

参数：
  - **input** (list) - 输入的LodTensorArray
  - **axis** (int) - 整数轴，tensor将会和它连接在一起
  - **name** (str|None) - 该layer的名字，可选。如果设置为none，layer将会被自动命名

返回：
    Variable: 连接的输出变量,输入LodTensorArray沿指定axis连接。

返回类型： Variable

**代码示例：**

.. code-block:: python

   import paddle.fluid as fluid
   tensor_array = fluid.layers.create_parameter(shape=[784, 200], dtype='float32')
   output, output_index = fluid.layers.tensor_array_to_tensor(input=tensor_array)











