.. _cn_api_fluid_layers_uniform_random:

uniform_random
-------------------------------

.. py:function:: paddle.fluid.layers.uniform_random(shape, dtype='float32', min=-1.0, max=1.0, seed=0)
该操作符初始化一个张量，该张量的值是从均匀分布中抽样的随机值

参数：
    - **shape** (LONGS)-输出张量的维
    - **dtype** (np.dtype|core.VarDesc.VarType|str) – 数据的类型, 例如float32, float64。 默认: float32.
    - **min** (FLOAT)-均匀随机分布的最小值。[默认 -1.0]
    - **max** (FLOAT)-均匀随机分布的最大值。[默认 1.0]
    - **seed** (INT)-随机种子，用于生成样本。0表示使用系统生成的种子。注意如果种子不为0，该操作符每次都生成同样的随机数。[默认 0]


**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    result = fluid.layers.uniform_random(shape=[32, 784])











