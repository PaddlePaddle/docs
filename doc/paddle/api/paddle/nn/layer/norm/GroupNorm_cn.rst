.. _cn_api_nn_GroupNorm:

GroupNorm
-------------------------------

.. py:class:: paddle.nn.GroupNorm(num_groups, num_channels, epsilon=1e-05, weight_attr=None, bias_attr=None, data_layout='NCHW, 'name=None)

**Group Normalization层**

该接口用于构建 ``GroupNorm`` 类的一个可调用对象，具体用法参照 ``代码示例`` 。其中实现了组归一化层的功能。更多详情请参考： `Group Normalization <https://arxiv.org/abs/1803.08494>`_ 。

参数：
    - **num_groups** (int) - 从通道中分离出来的 ``group`` 的数目。
    - **num_channels** (int) - 输入的通道数。
    - **epsilon** (float, 可选) - 为防止方差除零，增加一个很小的值。默认值：1e-05。
    - **weight_attr** (ParamAttr|bool, 可选) - 指定权重参数属性的对象。如果为False, 表示参数不学习。默认值为None，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
    - **bias_attr** (ParamAttr|bool, 可选) - 指定偏置参数属性的对象。如果为False, 表示参数不学习。默认值为None，表示使用默认的偏置参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
    - **data_format** (string, 可选) - 只支持“NCHW”(num_batches，channels，height，width)格式。默认值：“NCHW”。
    - **name** (string, 可选) – GroupNorm的名称, 默认值为None。更多信息请参见 :ref:`api_guide_Name` 。

返回：无

形状：
    - input: 形状为（批大小，通道数, 高度，宽度）的4-D Tensor。
    - output: 和输入形状一样。

**代码示例**

.. code-block:: python

   import paddle
   import numpy as np

   np.random.seed(123)
   x_data = np.random.random(size=(2, 6, 2, 2)).astype('float32')
   x = paddle.to_tensor(x_data) 
   group_norm = paddle.nn.GroupNorm(num_channels=6, num_groups=6)
   group_norm_out = group_norm(x)

   print(group_norm_out)
