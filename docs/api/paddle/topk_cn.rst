.. _cn_api_paddle_topk:

topk
-------------------------------

.. py:function:: paddle.topk(x, k, axis=None, largest=True, sorted=True, name=None)

沿着可选的 ``axis`` 查找 topk 最大或者最小的结果和结果所在的索引信息。
如果是一维 Tensor，则直接返回 topk 查询的结果。如果是多维 Tensor，则在指定的轴上查询 topk 的结果。

参数
:::::::::
    - **x** (Tensor) - 输入的多维 ``Tensor`` ，支持的数据类型：float32、float64、int32、int64。
    - **k** (int，Tensor) - 在指定的轴上进行 top 寻找的数量。
    - **axis** (int，可选) - 指定对输入 Tensor 进行运算的轴， ``axis`` 的有效范围是[-R, R），R 是输入 ``x`` 的 Rank， ``axis`` 为负时与 ``axis`` + R 等价。默认值为-1。
    - **largest** (bool，可选) - 指定算法排序的方向。如果设置为 True，排序算法按照降序的算法排序，否则按照升序排序。默认值为 True。
    - **sorted** (bool，可选) - 控制返回的结果是否按照有序返回，默认为 True。在 GPU 上总是返回有序的结果。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
tuple(Tensor)， 返回 topk 的结果和结果的索引信息。结果的数据类型和输入 ``x`` 一致。索引的数据类型是 int64。


代码示例
:::::::::

COPY-FROM: paddle.topk
