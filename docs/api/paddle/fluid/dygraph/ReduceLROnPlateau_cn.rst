.. _cn_api_fluid_dygraph_ReduceLROnPlateau:

ReduceLROnPlateau
-------------------------------

**注意：该 API 仅支持【动态图】模式**

.. py:class:: paddle.fluid.dygraph.ReduceLROnPlateau(learning_rate, mode='min', decay_rate=0.1, patience=10, verbose=False, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8, dtype='float32')

该 API 为 ``loss`` 自适应的学习率衰减策略。默认情况下，当 ``loss`` 停止下降时，降低学习率（如果将 ``mode`` 设置为 `'max'`，此时判断逻辑相反，``loss`` 停止上升时降低学习率）。其思想是：一旦模型表现不再提升，将学习率降低 2-10 倍对模型的训练往往有益。

``loss`` 是传入到该类方法 ``step`` 中的参数，其必须是 shape 为[]的 0-D Tensor。如果 ``loss`` 停止下降（``mode`` 为 `min` 时）超过 ``patience`` 个 epoch，学习率将会减小为
`learning_rate * decay_rate` 。

此外，每降低一次学习率后，将会进入一个时长为 ``cooldown`` 个 epoch 的冷静期，在冷静期内，将不会监控 ``loss`` 的变化情况，也不会衰减。
在冷静期之后，会继续监控 ``loss`` 的上升或下降。

参数
::::::::::::

    - **learning_rate** (Variable|float|int) - 初始学习率。其类型可以是 Python 的 float 类型，如果输入 int 类型则会被转为 float 类型。其也可以是 shape 为[1]的
      1-D Tensor，且相应数据类型必须为 "float32" 或 "float64" 。
    - **mode** (str，可选) - `'min'` 和 `'max'` 之一。通常情况下，为 `'min'`，此时当 ``loss`` 停止下降时学习率将减小。默认：`'min'` 。
      （注意：仅在特殊用法时，可以将其设置为 `'max'`，此时判断逻辑相反，``loss`` 停止上升学习率才减小）
    - **decay_rate** (float，可选) - 学习率衰减的比例。`new_lr = origin_lr * decay_rate`，它是值小于 1.0 的 float 型数字，默认：0.1。
    - **patience** (int，可选) - 当 ``loss`` 连续 ``patience`` 个 epoch 没有下降(mode: 'min')或上升(mode: 'max')时，学习率才会减小。默认：10。
    - **verbose** (bool，可选) - 如果为 ``True``，会在每次更新 optimizer 中的 learning_rate 时，打印信息。默认：``False`` 。
    - **threshold** (float，可选) - ``threshold`` 和 ``threshold_mode`` 两个参数将会决定 ``loss`` 最小变化的阈值。小于该阈值的变化
      将会被忽视。默认：1e-4。
    - **threshold_mode** (str，可选) - `'rel'` 和 `'abs'` 之一。在 `'rel'` 模式下，``loss`` 最小变化的阈值是 `last_loss * threshold` ，
      其中 ``last_loss`` 是 ``loss`` 在上个 epoch 的值。在 `'abs'` 模式下，``loss`` 最小变化的阈值是 `threshold`。默认：`'rel'`。
    - **cooldown** (int，可选) - 在学习速率每次减小之后，会进入时长为 ``cooldown`` 个 epoch 的冷静期。默认：0。
    - **min_lr** (float，可选) - 最小的学习率。减小后的学习率最低下界限。默认：0。
    - **eps** (float，可选) - 如果新旧学习率间的差异小于 ``eps``，则不会更新。默认值：1e-8。
    - **dtype** (str，可选) – 学习率值的数据类型，可以为"float32", "float64"。默认："float32"。

返回
::::::::::::
 ``loss`` 自适应的学习率

返回类型
::::::::::::
Variable

代码示例
::::::::::::

    .. code-block:: python

        import paddle.fluid as fluid
        import numpy as np

        with fluid.dygraph.guard():
            x = np.random.uniform(-1, 1, [10, 10]).astype("float32")
            linear = fluid.dygraph.Linear(10, 10)
            input = fluid.dygraph.to_variable(x)

            adam = fluid.optimizer.Adam(
                learning_rate = fluid.dygraph.ReduceLROnPlateau(
                                    learning_rate = 1.0,
                                    decay_rate = 0.5,
                                    patience = 5,
                                    verbose = True,
                                    cooldown = 3),
                parameter_list = linear.parameters())

            for epoch in range(10):
                total_loss = 0
                for bath_id in range(5):
                    out = linear(input)
                    loss = fluid.layers.reduce_mean(out)
                    total_loss += loss
                    adam.minimize(loss)

                avg_loss = total_loss/5

                # 根据传入 total_loss，调整学习率
                reduce_lr.step(avg_loss)
                lr = adam.current_step_lr()
                print("current avg_loss is %s, current lr is %s" % (avg_loss.numpy()[0], lr))



方法
::::::::::::
step(loss)
'''''''''
需要在每个 epoch 调用该方法，其根据传入的 ``loss`` 调整 optimizer 中的学习率，调整后的学习率将会在下一次调用 ``optimizer.minimize`` 时生效。

**参数**

  - **loss** (Variable) - 类型：Variable，shape 为[]的 0-D Tensor。将被用来判断是否需要降低学习率。如果 ``loss`` 连续 ``patience`` 个 epochs 没有下降，
    将会降低学习率。

**返回**

    无

**代码示例**

    参照其类中的说明。
