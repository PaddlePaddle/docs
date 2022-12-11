.. _cn_api_paddle_callbacks_EarlyStopping:

EarlyStopping
-------------------------------

.. py:class:: paddle.callbacks.EarlyStopping(monitor='loss', mode='auto', patience=0, verbose=1, min_delta=0, baseline=None, save_best_model=True)

在模型评估阶段，模型效果如果没有提升，``EarlyStopping`` 会通过设置 ``model.stop_training=True`` 让模型提前停止训练。

参数
::::::::::::

  - **monitor** (str，可选) - 监控量。该量作为模型是否停止学习的监控指标。默认值：'loss'。
  - **mode** (str，可选) - 可以是'auto'、'min'或者'max'。在 min 模式下，模型会在监控量的值不再减少时停止训练；max 模式下，模型会在监控量的值不再增加时停止训练；auto 模式下，实际的模式会从 ``monitor`` 推断出来。如果 ``monitor`` 中有'acc'，将会认为是 max 模式，其它情况下，都会被推断为 min 模式。默认值：'auto'。
  - **patience** (int，可选) - 多少个 epoch 模型效果未提升会使模型提前停止训练。默认值：0。
  - **verbose** (int，可选) - 可以是 0 或者 1，0 代表不打印模型提前停止训练的日志，1 代表打印日志。默认值：1。
  - **min_delta** (int|float，可选) - 监控量最小改变值。当 evaluation 的监控变量改变值小于 ``min_delta``，就认为模型没有变化。默认值：0。
  - **baseline** (int|float，可选) - 监控量的基线。如果模型在训练 ``patience`` 个 epoch 后效果对比基线没有提升，将会停止训练。如果是 None，代表没有基线。默认值：None。
  - **save_best_model** (bool，可选) - 是否保存效果最好的模型（监控量的值最优）。文件会保存在 ``fit`` 中传入的参数 ``save_dir`` 下，前缀名为 best_model，默认值：True。

代码示例
::::::::::::

COPY-FROM: paddle.callbacks.EarlyStopping
