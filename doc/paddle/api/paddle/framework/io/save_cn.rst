.. _cn_api_paddle_framework_io_save:

save
-----

.. py:function:: paddle.save(obj, model_path)

将对象实例obj保存到指定的路径中。

.. note::
    目前仅支持存储 Layer 或者 Optimizer 的 ``state_dict`` 。

参数:
 - **obj**  (Object) – 要保存的对象实例。
 - **path**  (str) – 保存对象实例的路径。如果存储到当前路径，输入的path字符串将会作为保存的文件名。

返回: 无
  
**代码示例**

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
