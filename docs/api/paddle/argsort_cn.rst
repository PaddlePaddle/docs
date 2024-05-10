.. _cn_api_paddle_argsort:

argsort
-------------------------------

.. py:function:: paddle.argsort(x, axis=-1, descending=False, stable=False, name=None)


对输入变量沿给定轴进行排序，输出排序好的数据的相应索引，其维度和输入相同。默认升序排列，如果需要降序排列设置 ``descending=True`` 。


参数
::::::::::::

    - **x** (Tensor) - 输入的多维 ``Tensor``，支持的数据类型：float16、bfloat16、float32、float64、int16、int32、int64、uint8。
    - **axis** (int，可选) - 指定对输入 Tensor 进行运算的轴，``axis`` 的有效范围是 [-R, R)，R 是输入 ``x`` 的 Rank， ``axis`` 为负时与 ``axis`` + R 等价。默认值为 -1。
    - **descending** (bool，可选) - 指定算法排序的方向。如果设置为 True，算法按照降序排序。如果设置为 False 或者不设置，按照升序排序。默认值为 False。
    - **stable** (bool，可选) - 是否使用稳定排序算法。若设置为 True，则使用稳定排序算法，即相同元素的顺序在排序结果中将会被保留。默认值为 False，此时的算法不一定是稳定排序算法。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
Tensor，排序后索引信息（与 ``x`` 维度信息一致），数据类型为 int64。


代码示例
::::::::::::

COPY-FROM: paddle.argsort
