.. _cn_api_paddle_compat_sort:

sort
-------------------------------

.. py:function:: paddle.compat.sort(input, dim=-1, descending=False, stable=False, out=None)

PyTorch 兼容的 :ref:`cn_api_paddle_sort` 版本，同时返回排序的值结果以及索引值。对输入变量沿给定轴进行排序，输出排序好的数据，其维度和输入相同。默认升序排列，如果需要降序排列设置 ``descending=True`` 。

使用前请详细参考：`【返回参数类型不一致】torch.sort`_ 以确定是否使用此模块。

.. _【返回参数类型不一致】torch.sort: https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/model_convert/convert_from_pytorch/api_difference/torch/torch.sort.html


参数
::::::::::::

    - **input** (Tensor) - 输入的多维 ``Tensor``，支持的数据类型：float32, float64, int16, int32, int64, uint8, float16, bfloat16。
    - **dim** (int，可选) - 指定对输入 Tensor 进行运算的轴，``dim`` 的有效范围是[-R, R)，R 是输入 ``x`` 的 Rank， ``dim`` 为负时与 ``dim`` +R 等价。默认值为-1。
    - **descending** (bool，可选) - 指定算法排序的方向。如果设置为 True，算法按照降序排序。如果设置为 False 或者不设置，按照升序排序。默认值为 False。
    - **stable** (bool，可选) - 是否使用稳定排序算法。若设置为 True，则使用稳定排序算法，即相同元素的顺序在排序结果中将会被保留。默认值为 False，此时的算法不一定是稳定排序算法。
    - **out** (tuple(Tensor, Tensor)，可选) - 用于引用式传入输出值。``values`` 在前，``indices`` 在后。注意：动态图下 out 可以是任意 Tensor，默认值为 None。``out`` 返回方法与静态图联合使用是被禁止的行为，静态图下将报错。

返回
::::::::::::
SortRetType(Tensor, Tensor)，此处的 ``SortRetType`` 是一个具名元组，含有 ``values`` （在前）和 ``indices`` （在后）两个域，用法与 tuple 一致。


代码示例
::::::::::::

COPY-FROM: paddle.compat.sort
