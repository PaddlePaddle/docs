.. _cn_api_paddle_histogramdd:

histogramdd
-------------------------------

.. py:function:: paddle.histogramdd(x, bins=10, ranges=None, density=False, weights=None, name=None)

计算输入多维 Tensor 的直方图。将最内维度大小为 N 的输入张量的元素解释为 N 维点的集合。将每个点映射到一组 N 维 bin 中，并返回每个 bin 中的点数（或总权重）

输入 x 必须是具有至少 2 个维度的张量。如果输入具有形状 （M，N） 则其 M 行中的每一行定义 N 维空间中的一个点。如果输入有三个或多个维度，则除最后一个维度外的所有维度都将被展平。


参数
::::::::::::

    - **input** (Tensor) - 输入多维 Tensor 。
    - **bins** (Tensor[]|int[]|int) - 如果为 Tensor 数组，则表示所有 bin 边界。如果为 int 数组，则表示每个维度中等宽 bin 的数量。如果为 int，则表示所有维度的等宽 bin 数量。默认值为 10 ，表示所有维度的等宽 bin 数量为 10 个。
    - **ranges** (float[], 可选) - 表示每个维度中最左边和最右边的 bin 边界。如果为 None ，则将每个尺寸的最小值和最大值设置为最左边和最右边。默认值为 None ，表示自动根据最大值与最小值计算 bin 的边界。
    - **density** (bool，可选) - 表示是否计算 density ，如果为 False，结果将包含每个 bin 中的计数（或权重）。如果为 True，则将每个计数（权重）除以总计数（总权重），然后再除以相关 bin 的宽度。默认值为 False ，表示不计算 density 。
    - **weights** (Tensor，可选) - 表示权重。如果传递 Tensor ，则输入中的每个 N 维坐标将其相关权重贡献给其 bin 的结果。权重应具有与输入张量相同的形状，但不包括其最内维度 N。默认情况下，输入中的每个值的权重为 1 。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
包含直方图值的 N 维张量，包含 bin 边界信息的 N 个 1D tensor 的序列。

代码示例
::::::::::::

COPY-FROM: paddle.histogramdd
