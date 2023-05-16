.. _cn_api_tensor_uniform:

uniform
-------------------------------

.. py:function:: paddle.uniform(shape, dtype=None, min=-1.0, max=1.0, seed=0, name=None)




返回数值服从范围[``min``, ``max``)内均匀分布的随机 Tensor，形状为 ``shape``，数据类型为 ``dtype``。

.. code-block:: text

    示例 1:
             给定：
                 shape=[1,2]
             则输出为：
                 result=[[0.8505902, 0.8397286]]

参数
::::::::::::

    - **shape** (list|tuple|Tensor) - 生成的随机 Tensor 的形状。如果 ``shape`` 是 list、tuple，则其中的元素可以是 int，或者是形状为[]且数据类型为 int32、int64 的 0-D Tensor。如果 ``shape`` 是 Tensor，则是数据类型为 int32、int64 的 1-D Tensor。
    - **dtype** (str|np.dtype，可选) - 输出 Tensor 的数据类型，支持 float32、float64。默认值为 None。
    - **min** (float|int，可选) - 要生成的随机值范围的下限，min 包含在范围中。支持的数据类型：float、int。默认值为-1.0。
    - **max** (float|int，可选) - 要生成的随机值范围的上限，max 不包含在范围中。支持的数据类型：float、int。默认值为 1.0。
    - **seed** (int，可选) - 随机种子，用于生成样本。0 表示使用系统生成的种子。注意如果种子不为 0，该操作符每次都生成同样的随机数。支持的数据类型：int。默认为 0。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::

Tensor：数值服从范围[``min``, ``max``)内均匀分布的随机 Tensor，形状为 ``shape``，数据类型为 ``dtype``。


代码示例
::::::::::::

COPY-FROM: paddle.uniform
