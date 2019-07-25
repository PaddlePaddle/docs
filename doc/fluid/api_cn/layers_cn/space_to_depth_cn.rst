.. _cn_api_fluid_layers_space_to_depth:

space_to_depth
-------------------------------

.. py:function:: paddle.fluid.layers.space_to_depth(x, blocksize, name=None)

给该函数一个 ``blocksize`` 值，可以对形为[batch, channel, height, width]的输入LoD张量进行space_to_depth（广度至深度）运算。

该运算对成块的空间数据进行重组，形成深度。确切地说，该运算输出一个输入LoD张量的拷贝，其高度，宽度维度上的值移动至通道维度上。

``blocksize`` 参数指明了数据块大小。

重组时，依据 ``blocksize`` , 生成形为 :math:`[batch, channel * blocksize * blocksize, height/blocksize, width/blocksize]` 的输出：

该运算适用于在卷积间重放缩激励函数，并保持所有的数据。

 - 在各位置上，不重叠的，大小为 :math:`block\_size * block\_size` 的块重组入深度depth
 - 输出张量的深度为 :math:`block\_size * block\_size * input\_channel`
 - 输入各个块中的Y,X坐标变为输出张量通道索引的高序部位
 - channel可以被blocksize的平方整除
 - 高度，宽度可以被blocksize整除

参数:
  - **x** (variable) – 输入LoD张量
  - **blocksize** (variable) – 在每个特征图上选择元素时采用的块大小，应该 > 2

返回：输出LoD tensor

返回类型：Variable

抛出异常：
  - ``TypeError`` - ``blocksize`` 必须是long类型

**代码示例**

..  code-block:: python

    import paddle.fluid as fluid
    import numpy as np

    data = fluid.layers.data(
        name='data', shape=[1, 4, 2, 2], dtype='float32', append_batch_size=False)
    space_to_depthed = fluid.layers.space_to_depth(
        x=data, blocksize=2)

    exe = fluid.Executor(fluid.CUDAPlace(0))
    data_np = np.arange(0,16).reshape((1,4,2,2)).astype('float32')
    out_main = exe.run(fluid.default_main_program(),
                  feed={'data': data_np},
                  fetch_list=[space_to_depthed])





