.. _cn_api_io_cn_WeightedRandomSampler:

WeightedRandomSampler
-------------------------------

.. py:class:: paddle.io.WeightedRandomSampler(weights, num_samples, replacement=True)

通过制定的权重随机采样，采样下标范围在 ``[0, len(weights) - 1]``，如果 ``replacement`` 为 ``True``，则下标可被采样多次

参数
:::::::::

    - **weights** (numpy.ndarray|paddle.Tensor|tuple|list) - 权重序列，需要是 numpy 数组，paddle.Tensor，list 或者 tuple 类型。
    - **num_samples** (int) - 采样样本数。
    - **replacement** (bool) - 是否采用有放回的采样，默认值为 True

返回
:::::::::
WeightedRandomSampler，返回根据权重随机采样下标的采样器



代码示例
:::::::::

COPY-FROM: paddle.io.WeightedRandomSampler
