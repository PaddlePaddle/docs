.. _cn_api_nn_functional_instance_norm:

instance_norm
-------------------------------

.. py:class:: paddle.nn.functional.instance_norm(x, running_mean=None, running_var=None, weight=None, bias=None, use_input_stats=True, momentum=0.9, eps=1e-05, data_format='NCHW', name=None)

推荐使用 :ref:`cn_api_nn_InstanceNorm1D`，:ref:`cn_api_nn_InstanceNorm2D`，:ref:`cn_api_nn_InstanceNorm3D`，由内部调用此方法。

参数
::::::::::::

    - **x** (Tensor) - 输入，数据类型为 float32, float64。
    - **running_mean** (Tensor，可选) - 均值的 Tensor。过时（已被删除，无法使用）
    - **running_var** (Tensor，可选) - 方差的 Tensor。过时（已被删除，无法使用）
    - **weight** (Tensor，可选) - 权重的 Tensor。默认值：None. 如果 weight 为 None 则 weight 被初始化为全 1 的 Tensor.
    - **bias** (Tensor，可选) - 偏置的 Tensor。默认值：None. 如果 bias 为 None 则 bias 被初始化为值等于 0 的 Tensor.
    - **use_input_stats** (bool，可选) - 默认是 True。过时（已被删除，无法使用）
    - **momentum** (float，可选) - 此值用于计算 ``moving_mean`` 和 ``moving_var``。默认值：0.9。更新公式如上所示。
    - **eps** (float，可选) - 为了数值稳定加在分母上的值。默认值：1e-05。
    - **data_format** (string，可选) - 指定输入数据格式，数据格式可以为“NC", "NCL", "NCHW" 或者"NCDHW"。默认值："NCHW"。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
无


代码示例
::::::::::::

COPY-FROM: paddle.nn.functional.instance_norm
