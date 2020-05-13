.. _cn_api_fluid_dygraph_ReduceLROnPlateau:
    
ReduceLROnPlateau
-------------------------------

**注意：该API仅支持【动态图】模式**

.. py:class:: paddle.fluid.dygraph.ReduceLROnPlateau(learning_rate, mode='min', decay_rate=0.1, patience=10, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=0, dtype='float32')

该接口为 ``loss`` 自适应的学习率衰减策略。当 ``loss`` 停止下降时，降低学习率。其思想是：一旦模型表现不再提升，将学习率降低2-10倍对模型的训练往往有益。

``loss`` 是传入到该类方法 ``step`` 中的参数，其必须是shape为[1]的1-D Tensor。 如果 ``loss`` 停止下降超过 ``patience`` 个epoch，学习率将会减小为
`learning_rate * decay_rate` 。

此外，每降低一次学习率后，重新恢复正常操作会等待 ``cooldown`` 个epoch，在该等待期间，将无视 ``loss`` 的变化情况。

参数：
    - **learning_rate** (Variable|float|int) - 初始学习率。其类型可以是Python的 float 或 int 类型。也可以是shape为[1]的
      1-D Tensor，且相应数据类型为"float32" 或 "float64"。
    - **mode** (str，可选) - `'min'` 和 `'max'` 之一。通常情况下，为 `'min'` ，此时当 ``loss`` 停止下降时学习率将减小。默认：`'min'` 。
      （注意：仅在特殊用法时，可以将其设置为 `'max'` ，此时判断逻辑相反， ``loss`` 停止上升学习率才减小）
    - **decay_rate** (float，可选) - 学习率衰减的比例。`new_lr = origin_lr * decay_rate` ，它是值小于1.0的float型数字，默认: 0.1。
    - **patience** (int，可选) - 当 ``loss`` 连续 ``patience`` 个epoch没有下降时，学习率才会减小。默认：10。
    - **threshold** (float，可选) - ``threshold`` 和 ``threshold_mode`` 两个参数将会决定 ``loss`` 最小变化的阈值。小于该阈值的变化
      将会被忽视。默认：1e-4。
    - **threshold_mode** (str，可选) - `'rel'` 和 `'abs'` 之一。在 `'rel'` 模式下， ``loss`` 最小变化的阈值是 `last_loss * threshold` ，
      其中 ``last_loss`` 是 ``loss`` 在上个epoch的值。在 `'abs'` 模式下，``loss`` 最小变化的阈值是 `threshold` 。 默认：`'rel'`。
    - **cooldown** (int，可选) - 在学习速率被降低之后，重新恢复正常操作之前等待的epoch数量。默认：0。
    - **min_lr** (float，可选) - 最小的学习率。减小后的学习率最低下界限。默认：0。
    - **dtype** (str，可选) – 学习率值的数据类型，可以为"float32", "float64"。默认："float32"。

返回： ``loss`` 自适应的学习率

返回类型：Variable

**代码示例**：

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
                
                # 根据传入total_loss，调整学习率
                reduce_lr.step(avg_loss)
                lr = adam.current_step_lr()
                print("current avg_loss is %s, current lr is %s" % (avg_loss.numpy()[0], lr))



.. py:method:: step(loss)
需要在每个epoch调用该方法，其根据传入的 ``loss`` 调整optimizer中的学习率，调整后的学习率将会在下一次调用 ``optimizer.minimize`` 时生效。

参数：
  - **loss** (Variable) - 类型：Variable，shape为[1]的1-D Tensor。将被用来判断是否需要降低学习率。如果 ``loss`` 连续 ``patience`` 个epochs没有下降，
    将会降低学习率。

返回：
    无

**代码示例**:

    参照其类中的说明。
