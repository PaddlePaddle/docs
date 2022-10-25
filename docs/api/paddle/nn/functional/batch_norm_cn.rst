.. _cn_api_nn_functional_batch_norm:

batch_norm
-------------------------------

.. py:class:: paddle.nn.functional.batch_norm(x, running_mean, running_var, weight, bias, training=False, momentum=0.9, epsilon=1e-05, data_format='NCHW', name=None):

推荐使用 nn.BatchNorm1D，nn.BatchNorm2D, nn.BatchNorm3D，由内部调用此方法。

详情见 :ref:`cn_api_nn_BatchNorm1D` 。

参数
::::::::::::

    - **x** (int) - 输入，数据类型为 float32, float64。
    - **running_mean** (Tensor) - 均值的 Tensor。
    - **running_var** (Tensor) - 方差的 Tensor。
    - **weight** (Tensor) - 权重的 Tensor。
    - **bias** (Tensor) - 偏置的 Tensor。
    - **momentum** (float，可选) - 此值用于计算 ``moving_mean`` 和 ``moving_var``。默认值：0.9。更新公式如上所示。
    - **epsilon** (float，可选) - 为了数值稳定加在分母上的值。默认值：1e-05。
    - **data_format** (string，可选) - 指定输入数据格式，数据格式可以为“NC", "NCL", "NCHW" 或者"NCDHW"。默认值："NCHW"。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。


返回
::::::::::::
无


代码示例
::::::::::::

COPY-FROM: paddle.nn.functional.batch_norm
