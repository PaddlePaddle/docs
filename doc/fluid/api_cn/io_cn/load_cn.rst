.. _cn_api_fluid_io_load:

load
-------------------------------

.. py:function:: paddle.fluid.io.load(program, model_path)

该接口尝试从磁盘中加载参数或优化器的dict。

该接口从Program中过滤出参数和优化器信息，然后从文件中获取相应的值。
如果程序和加载的文件之间参数的维度或数据类型不匹配，将引发异常。

**注意：此函数必须在运行启动程序（start_up_program）之后再调用。**

参数:
 - **program**  ( :ref:`cn_api_fluid_Program` ) – 要加载的Program。
 - **model_path**  (str) – 保存program的文件前缀。格式为 ``目录名称/文件前缀``。

返回: 两个dict，即从文件中恢复的参数dict和优化器dict

- para_dict: 从文件中恢复的参数dict
- opti_dict: 从文件中恢复的优化器dict

返回类型: tuple(dict, dict)

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid

    with fluid.dygraph.guard():
        emb = fluid.dygraph.Embedding( "emb", [10, 10])
        state_dict = emb.state_dict()
        fluid.save_dygraph( state_dict, "paddle_dy")  # 会保存为 paddle_dy.pdparams

        adam = fluid.optimizer.Adam( learning_rate = fluid.layers.noam_decay( 100, 10000) )
        state_dict = adam.state_dict()
        fluid.save_dygraph(state_dict, "paddle_dy")  # 会保存为 paddle_dy.pdopt

        para_state_dict, opti_state_dict = fluid.load_dygraph( "paddle_dy")  # 从磁盘加载参数和优化器的state_dict



