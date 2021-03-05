.. _cn_api_tensor_set_printoptions:

set_printoptions
-------------------------------

.. py:function:: paddle.set_printoptions(precision=None, threshold=None, edgeitems=None, sci_mode=None)



设置``paddle``中``Tensor``的打印配置选项。 注： 该函数与 ``numpy.set_printoptions()`` 类似。

参数
:::::::::
    - precision (int, 可选) - 浮点数的小数位数，默认值为8。
    - threshold (int, 可选) - 打印的元素个数上限，默认值为1000。
    - edgeitems (int, 可选) - 以缩略形式打印时左右两边的元素个数，默认值为3。
    - sci_mode (bool, 可选) - 是否以科学计数法打印，默认值为False。


返回
:::::::::
None


代码示例
:::::::::

..  code-block:: python

    import paddle

    paddle.seed(10)
    a = paddle.rand([10, 20])
    paddle.set_printoptions(4, 100, 3)
    print(a)
    
    '''
    Tensor(shape=[10, 20], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            [[0.0002, 0.8503, 0.0135, ..., 0.9508, 0.2621, 0.6661],
            [0.9710, 0.2605, 0.9950, ..., 0.4427, 0.9241, 0.9363],
            [0.0948, 0.3226, 0.9955, ..., 0.1198, 0.0889, 0.9231],
            ...,
            [0.7206, 0.0941, 0.5292, ..., 0.4856, 0.1379, 0.0351],
            [0.1745, 0.5621, 0.3602, ..., 0.2998, 0.4011, 0.1764],
            [0.0728, 0.7786, 0.0314, ..., 0.2583, 0.1654, 0.0637]])
    '''
