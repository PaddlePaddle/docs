.. _cn_api_nn_initializer_Assign:

Assign
-------------------------------

.. py:class:: paddle.nn.initializer.Assign(value, name=None)


该接口为参数初始化函数，使用 Numpy 数组、Python 列表、Tensor 来初始化参数。

参数
::::::::::::

    - **value** （Tensor|numpy.ndarray|list） - 用于初始化参数的一个 Numpy 数组、Python 列表、Tensor。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::

    由 Numpy 数组、Python 列表、Tensor 初始化的参数。

代码示例
::::::::::::

COPY-FROM: paddle.nn.initializer.Assign
