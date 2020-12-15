.. cn_api_nn_functional_instance_norm:

instance_norm
-------------------------------

.. py:class:: paddle.nn.functional.instance_norm(x, running_mean, running_var, weight, bias, training=False, epsilon=1e-05, momentum=0.9, use_input_stats=True, data_format='NCHW', name=None):

推荐使用nn.InstanceNorm1D，nn.InstanceNorm2D, nn.InstanceNorm3D，由内部调用此方法。

详情见 :ref:`cn_api_nn_InstanceNorm1D` 。

参数：
    - **x** (int) - 输入，数据类型为float32, float64。
    - **running_mean** (Tensor) - 均值的Tensor。
    - **running_var** (Tensor) - 方差的Tensor。
    - **weight** (Tensor) - 权重的Tensor。
    - **bias** (Tensor) - 偏置的Tensor。
    - **epsilon** (float, 可选) - 为了数值稳定加在分母上的值。默认值：1e-05。
    - **momentum** (float, 可选) - 此值用于计算 ``moving_mean`` 和 ``moving_var`` 。默认值：0.9。更新公式如上所示。
    - **use_input_stats** (bool, 可选) - 默认是True.
    - **data_format** (string, 可选) - 指定输入数据格式，数据格式可以为“NC", "NCL", "NCHW" 或者"NCDHW"。默认值："NCHW"。
    - **name** (string, 可选) – InstanceNorm的名称, 默认值为None。更多信息请参见 :ref:`api_guide_Name` 。

返回：无


**代码示例**

.. code-block:: python

    import paddle
    import numpy as np

    paddle.disable_static()
    x = np.random.seed(123)
    x = np.random.random(size=(2, 1, 2, 3)).astype('float32')
    running_mean = np.random.random(size=1).astype('float32')
    running_variance = np.random.random(size=1).astype('float32')
    weight_data = np.random.random(size=1).astype('float32')
    bias_data = np.random.random(size=1).astype('float32')
    x = paddle.to_tensor(x)
    rm = paddle.to_tensor(running_mean)
    rv = paddle.to_tensor(running_variance)
    w = paddle.to_tensor(weight_data)
    b = paddle.to_tensor(bias_data)
    instance_norm_out = paddle.nn.functional.instance_norm(x, rm, rv, w, b)
    print(instance_norm_out.numpy())
