.. _cn_api_fluid_dygraph_load_dygraph:

load
----


.. py:function:: paddle.load(model_path, config=None)

:api_attr: 命令式编程模式（动态图)

该接口用于从磁盘中加载Layer和Optimizer的 ``state_dict`` ，该接口会同时加载 ``model_path + ".pdparams"`` 和 ``model_path + ".pdopt"`` 中的内容。

.. note::
    由于一些历史原因，如果从 ``paddle.io.save_inference_model`` 的存储结果中载入 ``state_dict`` ，动态图模式下参数的结构性变量名将无法被恢复。并且在将载入的 ``state_dict`` 配置到当前Layer中时，需要配置 ``Layer.set_state_dict`` 的参数 ``use_structured_name=False`` 。

参数:
    - **model_path** (str) – 保存state_dict的文件前缀。该路径不应该包括后缀 ``.pdparams`` 或 ``.pdopt``。
    - **config** (SaveLoadConfig, 可选) - 用于指定额外配置选项的 :ref:`cn_api_fluid_dygraph_jit_SaveLoadConfig` 对象，这些选项主要是用于兼容 ``paddle.io.save_inference_model`` 存储模型的格式。默认为 ``None``。


返回: 两个 ``dict`` ，即从文件中恢复的模型参数 ``dict`` 和优化器参数 ``dict``，如果只找到其中一个的存储文件，另一个返回None

- param_dict: 从文件中恢复的模型参数 ``dict``
- opt_dict: 从文件中恢复的优化器参数 ``dict``

返回类型: tuple(dict, dict)
  
**代码示例**

.. code-block:: python

    import paddle
            
    paddle.disable_static()

    emb = paddle.nn.Embedding([10, 10])

    state_dict = emb.state_dict()
    paddle.save(state_dict, "paddle_dy")

    scheduler = paddle.optimizer.lr_scheduler.NoamLR(
        d_model=0.01, warmup_steps=100, verbose=True)
    adam = paddle.optimizer.Adam(
        learning_rate=scheduler,
        parameters=emb.parameters())
    state_dict = adam.state_dict()
    paddle.save(state_dict, "paddle_dy")

    para_state_dict, opti_state_dict = paddle.load("paddle_dy")



