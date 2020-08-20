.. _cn_api_optimizer_ReduceLROnPlateau

LambdaLR
-----------------------------------

.. py:class:: paddle.optimizer.lr_scheduler.ReduceLROnPlateau(learning_rate, mode='min', factor=0.1, patience=10, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=0, epsilon=1e-8, verbose=False)

该API为 ``loss`` 自适应的学习率衰减策略。默认情况下，当 ``loss`` 停止下降时，降低学习率（如果将 mode 设置为 'max' ，此时判断逻辑相反，loss 停止上升时降低学习率）。其思想是：一旦模型表现不再提升，将学习率降低2-10倍对模型的训练往往有益。
loss 是传入到该类方法 ``step`` 中的参数，其必须是shape为[1]的1-D Tensor。 如果 loss 停止下降（mode 为 min 时）超过 ``patience`` 个epoch，学习率将会减小为 learning_rate * factor。
此外，每降低一次学习率后，将会进入一个时长为 cooldown 个epoch的冷静期，在冷静期内，将不会监控 loss 的变化情况，也不会衰减。 在冷静期之后，会继续监控 loss 的上升或下降。


参数
:::::::::
    - **learning_rate** （float|int）：初始学习率，可以是Python的float或int。
    - **mode** （str，可选）'min' 和 'max' 之一。通常情况下，为 'min' ，此时当 loss 停止下降时学习率将减小。默认：'min' 。 （注意：仅在特殊用法时，可以将其设置为 'max' ，此时判断逻辑相反， loss 停止上升学习率才减小）
    - **fator** （float，可选） - 学习率衰减的比例。new_lr = origin_lr * factor，它是值小于1.0的float型数字，默认: 0.1。
    - **patience** （int，可选）- 当 loss 连续 patience 个epoch没有下降(mode: 'min')或上升(mode: 'max')时，学习率才会减小。默认：10。
    - **threshold** （float，可选）- threshold 和 threshold_mode 两个参数将会决定 loss 最小变化的阈值。小于该阈值的变化 将会被忽视。默认：1e-4。
    - **threshold_mode** （str，可选）- 'rel' 和 'abs' 之一。在 'rel' 模式下， loss 最小变化的阈值是 last_loss * threshold ， 其中 last_loss 是 loss 在上个epoch的值。在 'abs' 模式下，loss 最小变化的阈值是 threshold 。 默认：'rel'。
   - **cooldown** （int，可选）- 在学习速率每次减小之后，会进入时长为 ``cooldown`` 个 step 的冷静期。默认：0。
   - **min_lr** （float，可选） - 最小的学习率。减小后的学习率最低下界限。默认：0。
   - **epsilon** （float，可选）- 如果新旧学习率间的差异小于 eps ，则不会更新。默认值:1e-8。
    - **verbose** （bool）：如果是 `True` ，则在每一轮更新时在标准输出 `stdout` 输出一条信息。

返回
:::::::::
    返回计算ReduceLROnPlateau的可调用对象。

代码示例
:::::::::

.. code-block:: python



.. py:method:: step(loss)

需要在每个step调用该方法，其根据传入的 loss 调整optimizer中的学习率，调整后的学习率将会在下一个 ``step`` 时生效。

参数：
   loss (Tensor) - shape为[1]的1-D Tensor。将被用来判断是否需要降低学习率。如果 loss 连续 patience 个 ``steps`` 没有下降， 将会降低学习率。

返回：
    无

**代码示例**:

    参照上述示例代码。
