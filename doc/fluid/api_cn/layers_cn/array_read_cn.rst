.. _cn_api_fluid_layers_array_read:

array_read
-------------------------------

.. py:function:: paddle.fluid.layers.array_read(array,i)

该OP用于读取输入数组 :ref:`cn_api_fluid_LoDTensorArray` 中指定位置的数据, ``array`` 为输入的数组， ``i`` 为指定的读取位置。

可以类比为列表的元素读取：
::
    Given:
        array = [0.6,0.1,0.3,0.1]
    And:
        i = 2
    Then:
        output = 0.3

要注意的是，上述仅作为举例，该OP无法输入列表类型的数据。

参数：
    - **array** (Variable) - 输入的数组LoDTensorArray
    - **i** (Variable) - shape为[1]的1-D Tensor，表示从 ``array`` 中读取数据的位置，数据类型为int64


返回：从 ``array`` 中指定位置读取的LoDTensor

返回类型：Variable

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    array = fluid.layers.create_array(dtype='float32')
    i = fluid.layers.fill_constant(shape=[1],dtype='int64',value=10)
    item = fluid.layers.array_read(array, i)









