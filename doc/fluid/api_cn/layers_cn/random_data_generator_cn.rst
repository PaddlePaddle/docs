.. _cn_api_fluid_layers_random_data_generator:

random_data_generator
-------------------------------

.. py:function:: paddle.fluid.layers.random_data_generator(low, high, shapes, lod_levels, for_parallel=True)

创建一个均匀分布随机数据生成器.

该层返回一个Reader变量。该Reader变量不是用于打开文件读取数据，而是自生成float类型的均匀分布随机数。该变量可作为一个虚拟reader来测试网络，而不需要打开一个真实的文件。

参数：
    - **low** (float)--数据均匀分布的下界
    - **high** (float)-数据均匀分布的上界
    - **shapes** (list)-元组数列表，声明数据维度
    - **lod_levels** (list)-整形数列表，声明数据
    - **for_parallel** (Bool)-若要运行一系列操作命令则将其设置为True

返回：Reader变量，可从中获取随机数据

返回类型：变量(Variable)

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    reader = fluid.layers.random_data_generator(
                                 low=0.0,
                                 high=1.0,
                                 shapes=[[3,224,224], [1]],
                                 lod_levels=[0, 0])
    # 通过reader, 可以用'read_file'层获取数据:
    image, label = fluid.layers.read_file(reader)









