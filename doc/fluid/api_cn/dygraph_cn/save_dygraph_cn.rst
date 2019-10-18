.. _cn_api_fluid_dygraph_save_dygraph:

save_dygraph
-------------------------------

.. py:function:: paddle.fluid.dygraph.save_dygraph(model_dict, dirname='save_dir', optimizers=None)

该接口将传入的参数或优化器的dict保存到磁盘上。

``state_dict`` 是通过 :ref:`cn_api_fluid_dygraph_Layer` 的 ``state_dict()`` 方法得到的。

参数:
 - **state_dict**  (dict of Parameters) – 要保存的模型参数的dict。
 - **model_path**  (str) – 保存state_dict的文件前缀。格式为 ``目录名称/文件前缀``。如果文件前缀为空字符串，会引发异常。

返回: None
  
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
    
    





