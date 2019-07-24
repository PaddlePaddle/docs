.. _cn_api_fluid_layers_gaussian_random:

gaussian_random
-------------------------------

.. py:function:: paddle.fluid.layers.gaussian_random(shape, mean=0.0, std=1.0, seed=0, dtype='float32')

gaussian_random算子。

用于使用高斯随机生成器初始化张量（Tensor）。

参数：
        - **shape** （tuple | list）- （vector <int>）随机张量的维数
        - **mean** （Float）- （默认值0.0）随机张量的均值
        - **std** （Float）- （默认值为1.0）随机张量的std
        - **seed** （Int）- （默认值为 0）生成器随机生成种子。0表示使用系统范围的种子。注意如果seed不为0，则此算子每次将始终生成相同的随机数
        - **dtype** （np.dtype | core.VarDesc.VarType | str）- 输出的数据类型。

返回：        输出高斯随机运算矩阵

返回类型：        输出（Variable）

**代码示例：**

.. code-block:: python

    import paddle.fluid as fluid
    import paddle.fluid.layers as layers
    out = fluid.layers.gaussian_random(shape=[20, 30])








