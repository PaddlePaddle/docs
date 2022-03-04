.. _cn_api_tensor_outer:

outer
-------------------------------

.. py:function:: paddle.outer(x, y, name=None)


计算两个Tensor的外积。

对于1维Tensor正常计算外积，对于大于1维的Tensor先展平为1维再计算外积。

参数：
:::::::::
    - **x** (Tensor) - 一个N维Tensor或者标量Tensor。
    - **y** (Tensor) - 一个N维Tensor或者标量Tensor。
    - **name** (str, 可选) - 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回：
:::::::::
    - Tensor, x、y的外积结果，Tensor shape为 [x.size, y.size]。

代码示例：
::::::::::

.. code-block:: python
    
    import paddle

    x = paddle.arange(1, 4).astype('float32')
    y = paddle.arange(1, 6).astype('float32')
    out = paddle.outer(x, y)
    
    print(out)
    #        ([[1, 2, 3, 4, 5],
    #         [2, 4, 6, 8, 10],
    #         [3, 6, 9, 12, 15]])
