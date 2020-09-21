.. _cn_api_paddle_framework_io_load:

load
-----

.. py:function:: paddle.load(path, config=None)

从指定路径载入可以在paddle中使用的对象实例。

.. note::
    目前仅支持载入 Layer 或者 Optimizer 的 ``state_dict`` 。

.. note::
    ``paddle.load`` 支持从paddle1.x版本中静态图save相关API的存储结果中载入 ``state_dict`` ，但由于一些历史原因，如果从 ``paddle.static.save_inference_model/paddle.fluid.io.save_params/paddle.fluid.io.save_persistables`` 的存储结果中载入 ``state_dict`` ，动态图模式下参数的结构性变量名将无法被恢复。并且在将载入的 ``state_dict`` 配置到当前Layer中时，需要配置 ``Layer.set_state_dict`` 的参数 ``use_structured_name=False`` 。

参数:
    - **path** (str) – 载入目标对象实例的路径。通常该路径是目标文件的路径，在兼容载入 ``paddle.jit.save/paddle.static.save_inference_model`` 的存储结果时，该路径是一个目录。
    - **config** (SaveLoadConfig, 可选) - 用于指定额外配置选项的 :ref:`cn_api_fluid_dygraph_jit_SaveLoadConfig` 对象，这些选项主要是用于兼容 ``paddle.jit.save/paddle.static.save_inference_model`` 存储结果的格式。默认为 ``None``。


返回: 一个可以在paddle中使用的对象实例

返回类型: Object
  
**代码示例**

.. code-block:: python

    import paddle
            
    paddle.disable_static()

    emb = paddle.nn.Embedding([10, 10])
    layer_state_dict = emb.state_dict()
    paddle.save(layer_state_dict, "emb.pdparams")

    adam = paddle.optimizer.Adam(
        learning_rate=0.001,
        parameters=emb.parameters())
    opt_state_dict = adam.state_dict()
    paddle.save(opt_state_dict, "adam.pdopt")

    load_layer_state_dict = paddle.load("emb.pdparams")
    load_opt_state_dict = paddle.load("adam.pdopt")
