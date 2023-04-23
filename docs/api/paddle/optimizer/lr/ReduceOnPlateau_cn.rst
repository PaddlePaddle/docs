.. _cn_api_paddle_optimizer_lr_ReduceOnPlateau:

ReduceOnPlateau
-----------------------------------

.. py:class:: paddle.optimizer.lr.ReduceOnPlateau(learning_rate, mode='min', factor=0.1, patience=10, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=0, epsilon=1e-8, verbose=False)

`loss` 自适应的学习率衰减策略。默认情况下，当 ``loss`` 停止下降时，降低学习率。其思想是：一旦模型表现不再提升，将学习率降低 2-10 倍对模型的训练往往有益。

`loss` 是传入到该类方法 ``step`` 中的 ``metrics`` 参数，其可以是 float 或者 shape 为[1]的 Tensor 或 numpy\.ndarray。如果 loss 停止下降超过 ``patience`` 个 epoch，学习率将会衰减为 ``learning_rate * factor`` （特殊地，``mode`` 也可以被设置为 ``'max'``，此时逻辑相反）。

此外，每降低一次学习率后，将会进入一个时长为 ``cooldown`` 个 epoch 的冷静期，在冷静期内，将不会监控 ``loss`` 的变化情况，也不会衰减。在冷静期之后，会继续监控 ``loss`` 的上升或下降。


参数
::::::::::::

    - **learning_rate** (float) - 初始学习率，数据类型为 Python float。
    - **mode** (str，可选) - ``'min'`` 和 ``'max'`` 之一。通常情况下，为 ``'min'``，此时当 ``loss`` 停止下降时学习率将衰减。默认：``'min'`` 。 （注意：仅在特殊用法时，可以将其设置为 ``'max'``，此时判断逻辑相反，``loss`` 停止上升学习率才衰减）。
    - **factor** (float，可选) - 学习率衰减的比例。``new_lr = origin_lr * factor``，它是值小于 1.0 的 float 型数字，默认：0.1。
    - **patience** (int，可选) - 当 ``loss`` 连续 ``patience`` 个 epoch 没有下降(对应 mode: 'min')或上升(对应 mode: 'max')时，学习率才会衰减。默认：10。
    - **threshold** (float，可选) - ``threshold`` 和 ``threshold_mode`` 两个参数将会决定 ``loss`` 最小变化的阈值。小于该阈值的变化将会被忽视。默认：1e-4。
    - **threshold_mode** (str，可选) - ``'rel'`` 和 ``'abs'`` 之一。在 ``'rel'`` 模式下，``loss`` 最小变化的阈值是 ``last_loss * threshold``，其中 ``last_loss`` 是 ``loss`` 在上个 epoch 的值。在 ``'abs'`` 模式下，``loss`` 最小变化的阈值是 ``threshold``。默认：``'rel'`` 。
    - **cooldown** (int，可选) - 在学习率每次衰减之后，会进入时长为 ``cooldown`` 个 step 的冷静期。默认：0。
    - **min_lr** (float，可选) - 最小的学习率。衰减后的学习率最低下界限。默认：0。
    - **epsilon** (float，可选) - 如果新旧学习率间的差异小于 epsilon，则不会更新。默认值：1e-8。
    - **verbose** (bool，可选) - 如果是 `True`，则在每一轮更新时在标准输出 `stdout` 输出一条信息。默认值为 ``False`` 。

返回
::::::::::::
用于调整学习率的 ``ReduceOnPlateau`` 实例对象。

代码示例
::::::::::::

COPY-FROM: paddle.optimizer.lr.ReduceOnPlateau

方法
::::::::::::
step(metrics, epoch=None)
'''''''''

step 函数需要在优化器的 `optimizer.step()` 函数之后调用，其根据传入的 metrics 调整 optimizer 中的学习率，调整后的学习率将会在下一个 ``step`` 时生效。

**参数**

  - **metrics** (Tensor|numpy.ndarray|float）- 用来判断是否需要降低学习率。如果 ``loss`` 连续 ``patience`` 个 ``step`` 没有下降，将会降低学习率。可以是 Tensor 或者 numpy.array，但是 shape 必须为[1]，也可以是 Python 的 float 类型。
  - **epoch** (int，可选) - 指定具体的 epoch 数。默认值 None，此时将会从-1 自动累加 ``epoch`` 数。

**返回**

无。

**代码示例**

参照上述示例代码。
