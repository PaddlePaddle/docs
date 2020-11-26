.. _cn_api_fluid_layers_shape:

shape
-------------------------------

.. py:function:: paddle.fluid.layers.shape(input)




shape层。

获得输入Tensor或SelectedRows的shape。

::

    示例1:
        输入是 N-D Tensor类型:
            input = [ [1, 2, 3, 4], [5, 6, 7, 8] ]

        输出shape:
            input.shape = [2, 4]

    示例2:
        输入是 SelectedRows类型:
            input.rows = [0, 4, 19]
            input.height = 20
            input.value = [ [1, 2], [3, 4], [5, 6] ]  # inner tensor
        输出shape:
            input.shape = [3, 2]

参数：
        - **input** （Tensor）-  输入的多维Tensor或SelectedRows，数据类型为float16，float32，float64，int32，int64。如果输入是SelectedRows类型，则返回其内部持有Tensor的shape。


返回： 一个Tensor，表示输入Tensor或SelectedRows的shape。

返回类型： Tensor。

**代码示例：**

.. code-block:: python

    import paddle
    x = paddle.randn((2,3))
    print(paddle.shape(x)) # [2, 3]
    