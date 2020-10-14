.. _cn_api_paddle_framework_io_save:

save
-----

.. py:function:: paddle.save(obj, path)

将对象实例obj保存到指定的路径中。

.. note::
    目前仅支持存储 Layer 或者 Optimizer 的 ``state_dict`` 。

.. note::
    不同于 ``paddle.jit.save`` ，由于 ``paddle.save`` 的存储结果是单个文件，所以不需要通过添加后缀的方式区分多个存储文件，``paddle.save`` 的输入参数 ``path`` 将直接作为存储结果的文件名而非前缀。为了统一存储文件名的格式，我们推荐使用paddle标椎文件后缀：
    1. 对于 ``Layer.state_dict`` ，推荐使用后缀 ``.pdparams`` ；
    2. 对于 ``Optimizer.state_dict`` ，推荐使用后缀 ``.pdopt`` 。
    具体示例请参考API的代码示例。

参数
:::::::::
 - **obj**  (Object) – 要保存的对象实例。
 - **path**  (str) – 保存对象实例的路径。如果存储到当前路径，输入的path字符串将会作为保存的文件名。

返回
:::::::::
无

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
