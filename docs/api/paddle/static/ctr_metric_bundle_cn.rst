.. _cn_api_paddle_static_ctr_metric_bundle:

ctr_metric_bundle
-------------------------------

.. py:function:: paddle.static.ctr_metric_bundle(input, label, ins_tag_weight=None)

CTR 相关度量层

此函数用于计算 CTR 相关指标：RMSE（均方根误差）、MAE（平均绝对误差）、predicted_ctr（预测点击率）、q 值。

为了计算这些指标的最终值，我们应该使用总实例数进行以下计算：

    - MAE = local_abserr / 实例数
    - RMSE = sqrt(local_sqrerr / 实例数)
    - predicted_ctr = local_prob / 实例数
    - q = local_q / 实例数

注意，如果您正在进行分布式作业，您应该首先对这些指标和实例数进行全局归约。

参数
::::::::::::
    - **input** (Tensor) - 一个浮点数 2D 张量，值在[0, 1]范围内。每行按降序排列。这个输入应该是 topk 的输出。通常，这个张量表示每个标签的概率。
    - **label** (Tensor) - 表示训练数据标签的 2D 整数张量。高度为批量大小，宽度始终为 1。
    - **ins_tag_weight** (Tensor) - 表示训练数据的 ins_tag_weight 的 2D 整数张量。1 表示真实数据，0 表示假数据。类型为 float32 或 float64 的 DenseTensor 或 Tensor。

返回
::::::::::::
    - **local_sqrerr** (Tensor) - 局部平方误差和
    - **local_abserr** (Tensor) - 局部绝对误差和
    - **local_prob** (Tensor) - 局部预测 CTR 和
    - **local_q** (Tensor) - 局部 q 值和
    - **local_pos_num** (Tensor) - 局部正例数
    - **local_ins_num** (Tensor) - 局部样本数

    tuple (local_sqrerr, local_abserr, local_prob, local_q, local_pos_num, local_ins_num): 包含局部平方误差和、局部绝对误差和、局部预测 CTR 和、局部 q 值和、局部正例数和局部样本数的元组。


代码示例：
::::::::::

COPY-FROM: paddle.static.ctr_metric_bundle
