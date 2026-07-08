.. _cn_api_paddle_nn_functional_instance_norm:

instance_norm
-------------------------------

.. py:function:: paddle.nn.functional.instance_norm(x, running_mean=None, running_var=None, weight=None, bias=None, use_input_stats=True, momentum=0.9, eps=1e-05, data_format='NCHW', name=None)

.. note::
    此 API 有两种调用方式：
    1. ``paddle.nn.functional.instance_norm(x, running_mean=None, running_var=None, weight=None, bias=None, use_input_stats=True, momentum=0.9, eps=1e-05, data_format='NCHW', name=None)`` (Paddle 风格)。
    2. ``paddle.nn.functional.instance_norm(input, running_mean=None, running_var=None, weight=None, bias=None, use_input_stats=True, momentum=0.1, eps=1e-05)`` (PyTorch 风格)。

推荐使用 :ref:`cn_api_paddle_nn_InstanceNorm1D`，:ref:`cn_api_paddle_nn_InstanceNorm2D`，:ref:`cn_api_paddle_nn_InstanceNorm3D`，由内部调用此方法。

参数
::::::::::::

    - **x** (Tensor) - 输入，数据类型为 float32, float64。别名 ``input``。
    - **running_mean** (Tensor，可选) - 运行均值。默认值：None。过时（已被删除，无法使用）
    - **running_var** (Tensor，可选) - 运行方差。默认值：None。过时（已被删除，无法使用）
    - **weight** (Tensor，可选) - instance_norm 权重的 Tensor。默认值：None。如果 weight 为 None 则 weight 被初始化为全 1 的 Tensor。
    - **bias** (Tensor，可选) - instance_norm 偏置的 Tensor。默认值：None。如果 bias 为 None 则 bias 被初始化为值等于 0 的 Tensor。
    - **eps** (float，可选) - 为了数值稳定加在分母上的值。默认值：1e-5。
    - **momentum** (float，可选) - 此值用于计算 ``moving_mean`` 和 ``moving_var``。默认值：0.9。
    - **use_input_stats** (bool，可选) - 默认值是 True。过时（已被删除，无法使用）
    - **data_format** (str，可选) - 指定输入数据格式，数据格式可以为 "NC", "NCL", "NCHW" 或者 "NCDHW"。默认值："NCHW"。
    - **name** (str|None，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
无


代码示例
::::::::::::

COPY-FROM: paddle.nn.functional.instance_norm
