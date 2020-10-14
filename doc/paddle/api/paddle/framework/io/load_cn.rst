.. _cn_api_paddle_framework_io_load:

load
-----

.. py:function:: paddle.load(path, **configs)

从指定路径载入可以在paddle中使用的对象实例。

.. note::
    目前仅支持载入 Layer 或者 Optimizer 的 ``state_dict`` 。

.. note::
    为了更高效地使用paddle存储的模型参数， ``paddle.load`` 支持从除 ``paddle.save`` 之外的其他save相关API的存储结果中载入 ``state_dict`` ，但是在不同场景中，参数 ``path`` 的形式有所不同：
    1. 从 ``paddle.static.save`` 或者 ``paddle.Model().save(training=True)`` 的保存结果载入： ``path`` 需要是完整的文件名，例如 ``model.pdparams`` 或者 ``model.opt`` ； 
    2. 从 ``paddle.jit.save`` 或者 ``paddle.static.save_inference_model`` 或者 ``paddle.Model().save(training=False)`` 的保存结果载入： ``path`` 需要是路径前缀， 例如 ``model/mnist`` ， ``paddle.load`` 会从 ``mnist.pdmodel`` 和 ``mnist.pdiparams`` 中解析 ``state_dict`` 的信息并返回。
    3. 从paddle 1.x API ``paddle.fluid.io.save_inference_model`` 或者 ``paddle.fluid.io.save_params/save_persistables`` 的保存结果载入： ``path`` 需要是目录，例如 ``model`` ，此处model是一个文件夹路径。

.. note::
   如果从 ``paddle.static.save`` 或者 ``paddle.static.save_inference_model`` 等静态图API的存储结果中载入 ``state_dict`` ，动态图模式下参数的结构性变量名将无法被恢复。在将载入的 ``state_dict`` 配置到当前Layer中时，需要配置 ``Layer.set_state_dict`` 的参数 ``use_structured_name=False`` 。

参数
:::::::::
    - path (str) – 载入目标对象实例的路径。通常该路径是目标文件的路径，当从用于存储预测模型API的存储结果中载入state_dict时，该路径可能是一个文件前缀或者目录。
    - **config (dict, 可选) - 其他用于兼容的载入配置选项。这些选项将来可能被移除，如果不是必须使用，不推荐使用这些配置选项。默认为 ``None``。目前支持以下配置选项：(1) model_filename (str) - paddle 1.x版本 ``save_inference_model`` 接口存储格式的预测模型文件名，原默认文件名为 ``__model__`` ； (2) params_filename (str) - paddle 1.x版本 ``save_inference_model`` 接口存储格式的参数文件名，没有默认文件名，默认将各个参数分散存储为单独的文件。

返回
:::::::::
Object，一个可以在paddle中使用的对象实例
  
代码示例
:::::::::

.. code-block:: python

    import paddle
            
    paddle.disable_static()

    emb = paddle.nn.Embedding(10, 10)
    layer_state_dict = emb.state_dict()
    paddle.save(layer_state_dict, "emb.pdparams")

    scheduler = paddle.optimizer.lr_scheduler.NoamLR(
        d_model=0.01, warmup_steps=100, verbose=True)
    adam = paddle.optimizer.Adam(
        learning_rate=scheduler,
        parameters=emb.parameters())
    opt_state_dict = adam.state_dict()
    paddle.save(opt_state_dict, "adam.pdopt")

    load_layer_state_dict = paddle.load("emb.pdparams")
    load_opt_state_dict = paddle.load("adam.pdopt")
