.. _cn_api_fluid_layers_sequence_enumerate:

sequence_enumerate
-------------------------------


.. py:function:: paddle.fluid.layers.sequence_enumerate(input, win_size, pad_value=0, name=None)

:api_attr: 声明式编程模式（静态图)



枚举形状为 ``[d_1, 1]`` 的输入序列所有长度为 ``win_size`` 的子序列，生成一个形状为 ``[d_1, win_size]`` 的新序列，需要时以 ``pad_value`` 填充。

注意,该OP的输入 ``input`` 只能是LodTensor。

范例如下：

::

        给定输入 x：
            x.lod =  [[0,            3,      5]]
            x.data = [[1], [2], [3], [4], [5]]  
            x.dims = [5, 1]
        设置属性 win_size = 2  pad_value = 0
        
        得到输出 out：
            out.lod =  [[0,                     3,            5]]  
            out.data = [[1, 2], [2, 3], [3, 0], [4, 5], [5, 0]]  
            out.dims = [5, 2]

参数：
        - **input** （Variable）- 输入序列，形状为 ``[d_1, 1]`` ，lod level为1的LodTensor。数据类型支持int32，int64，float32或float64。
        - **win_size** （int）- 子序列窗口大小。
        - **pad_value** （int，可选）- 填充值，默认为0。
        - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回：      枚举序列，形状为 ``[d_1, win_size]`` ，lod_level为1的LoDTensor。数据类型与输入 ``input`` 一致。

返回类型：   Variable

**代码示例**

..  code-block:: python

      import paddle.fluid as fluid
      x = fluid.layers.data(name='x',shape=[-1, 1], dtype='int32', lod_level=1)
      out = fluid.layers.sequence_enumerate(input=x, win_size=3, pad_value=0)









