.. _cn_api_nn_initializer_Assign:

Assign
-------------------------------

.. py:class:: paddle.nn.initializer.Assign(value, name=None)


该OP使用Numpy数组、Python列表、Tensor来初始化参数。

参数
::::::::::::

    - **value** （Tensor|numpy.ndarray|list） - 用于初始化参数的一个Numpy数组、Python列表、Tensor。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::

    由Numpy数组、Python列表、Tensor初始化的参数。

代码示例
::::::::::::

COPY-FROM: paddle.nn.initializer.Assign
