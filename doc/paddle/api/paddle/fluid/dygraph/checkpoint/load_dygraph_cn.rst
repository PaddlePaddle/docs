.. _cn_api_fluid_dygraph_load_dygraph:

load_dygraph
-------------------------------


.. py:function:: paddle.fluid.dygraph.load_dygraph(model_path)

:api_attr: 命令式编程模式（动态图)



该接口尝试从磁盘中加载参数或优化器的 ``dict`` 。

该接口会同时加载 ``model_path + ".pdparams"`` 和 ``model_path + ".pdopt"`` 中的内容。

参数:
    - **model_path**  (str) – 保存state_dict的文件前缀。该路径不应该包括后缀 ``.pdparams`` 或 ``.pdopt``。


返回: 两个 ``dict`` ，即从文件中恢复的参数 ``dict`` 和优化器 ``dict``

- para_dict: 从文件中恢复的参数 ``dict``
- opti_dict: 从文件中恢复的优化器 ``dict``

返回类型: tuple(dict, dict)
  
**代码示例**

.. code-block:: python

    import paddle.fluid as fluid

    with fluid.dygraph.guard():
        emb = fluid.dygraph.Embedding([10, 10])
        state_dict = emb.state_dict()
        fluid.save_dygraph( state_dict, "paddle_dy")
        adam = fluid.optimizer.Adam( learning_rate = fluid.layers.noam_decay( 100, 10000) ,
                                     parameter_list = emb.parameters() )
        state_dict = adam.state_dict()
        fluid.save_dygraph( state_dict, "paddle_dy")

        para_state_dict, opti_state_dict = fluid.load_dygraph( "paddle_dy")



