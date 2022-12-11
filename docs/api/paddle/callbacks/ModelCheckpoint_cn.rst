.. _cn_api_paddle_callbacks_ModelCheckpoint:

ModelCheckpoint
-------------------------------

.. py:class:: paddle.callbacks.ModelCheckpoint(save_freq=1, save_dir=None)

 ``ModelCheckpoint`` 回调类和 model.fit 联合使用，在训练阶段，保存模型权重和优化器状态信息。当前仅支持在固定的 epoch 间隔保存模型，不支持按照 batch 的间隔保存。

   子方法可以参考基类。

参数
::::::::::::

  - **save_freq** (int，可选) - 间隔多少个 epoch 保存模型。默认值：1。
  - **save_dir** (int，可选) - 保存模型的文件夹。如果不设定，将不会保存模型。默认值：None。


代码示例
::::::::::::

COPY-FROM: paddle.callbacks.ModelCheckpoint
