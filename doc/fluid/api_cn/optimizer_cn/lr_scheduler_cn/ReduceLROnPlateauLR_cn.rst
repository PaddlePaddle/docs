.. _cn_api_paddle_optimizer_ReduceLROnPlateau:

ReduceLROnPlateau
-----------------------------------

.. py:class:: paddle.optimizer.lr_scheduler.ReduceLROnPlateau(learning_rate, mode='min', factor=0.1, patience=10, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=0, epsilon=1e-8, verbose=False)

该API为 ``loss`` 自适应的学习率衰减策略。默认情况下，当 ``loss`` 停止下降时，降低学习率（如果将 mode 设置为 'max' ，此时判断逻辑相反，loss 停止上升时降低学习率）。其思想是：一旦模型表现不再提升，将学习率降低2-10倍对模型的训练往往有益。
loss 是传入到该类方法 ``step`` 中的参数，其必须是shape为[1]的1-D Tensor。 如果 loss 停止下降（mode 为 min 时）超过 ``patience`` 个epoch，学习率将会减小为 learning_rate * factor。
此外，每降低一次学习率后，将会进入一个时长为 cooldown 个epoch的冷静期，在冷静期内，将不会监控 loss 的变化情况，也不会衰减。 在冷静期之后，会继续监控 loss 的上升或下降。


参数
:::::::::
    - **learning_rate** （float） - 初始学习率，数据类型为Python float。
    - **mode** （str，可选）'min' 和 'max' 之一。通常情况下，为 'min' ，此时当 loss 停止下降时学习率将减小。默认：'min' 。 （注意：仅在特殊用法时，可以将其设置为 'max' ，此时判断逻辑相反， loss 停止上升学习率才减小）
    - **factor** （float，可选） - 学习率衰减的比例。new_lr = origin_lr * factor，它是值小于1.0的float型数字，默认: 0.1。
    - **patience** （int，可选）- 当 loss 连续 patience 个epoch没有下降(mode: 'min')或上升(mode: 'max')时，学习率才会减小。默认：10。
    - **threshold** （float，可选）- threshold 和 threshold_mode 两个参数将会决定 loss 最小变化的阈值。小于该阈值的变化 将会被忽视。默认：1e-4。
    - **threshold_mode** （str，可选）- 'rel' 和 'abs' 之一。在 'rel' 模式下， loss 最小变化的阈值是 last_loss * threshold ， 其中 last_loss 是 loss 在上个epoch的值。在 'abs' 模式下，loss 最小变化的阈值是 threshold 。 默认：'rel'。
   - **cooldown** （int，可选）- 在学习速率每次减小之后，会进入时长为 ``cooldown`` 个 step 的冷静期。默认：0。
   - **min_lr** （float，可选） - 最小的学习率。减小后的学习率最低下界限。默认：0。
   - **epsilon** （float，可选）- 如果新旧学习率间的差异小于epsilon ，则不会更新。默认值:1e-8。
    - **verbose** （bool，可选）：如果是 `True` ，则在每一轮更新时在标准输出 `stdout` 输出一条信息。默认值为 ``False`` 。

返回
:::::::::
    返回计算ReduceLROnPlateau的可调用对象。

代码示例
:::::::::

.. code-block:: python

    import paddle
    import numpy as np

    # train on default dygraph mode
    paddle.disable_static()
    x = np.random.uniform(-1, 1, [10, 10]).astype("float32")
    linear = paddle.nn.Linear(10, 10)
    scheduler = paddle.optimizer.lr_scheduler.ReduceLROnPlateau(learning_rate=1.0, factor=0.5, patience=5, verbose=True)
    sgd = paddle.optimizer.SGD(learning_rate=scheduler, parameter_list=linear.parameters())
    for epoch in range(20):
        for batch_id in range(2):
            x = paddle.to_tensor(x)
            out = linear(x)
            loss = paddle.reduce_mean(out)
            loss.backward()
            sgd.minimize(loss)
            linear.clear_gradients()
        scheduler.step(loss)

    # train on static mode
    paddle.enable_static()
    main_prog = paddle.static.Program()
    start_prog = paddle.static.Program()
    with paddle.static.program_guard(main_prog, start_prog):
        x = paddle.static.data(name='x', shape=[None, 4, 5])
        y = paddle.static.data(name='y', shape=[None, 4, 5])
        z = paddle.static.nn.fc(x, 100)
        loss = paddle.mean(z)
        scheduler = paddle.optimizer.lr_scheduler.ReduceLROnPlateau(learning_rate=1.0, factor=0.5, patience=5, verbose=True)
        sgd = paddle.optimizer.SGD(learning_rate=scheduler)
        sgd.minimize(loss)

    exe = paddle.static.Executor()
    exe.run(start_prog)
    for epoch in range(20):
        for batch_id in range(2):
            out = exe.run(
                main_prog,
                feed={
                    'x': np.random.randn(3, 4, 5).astype('float32'),
                    'y': np.random.randn(3, 4, 5).astype('float32')
                },
                fetch_list=loss.name)
        scheduler.step(out[0])


.. py:method:: step(metrics, epoch=None) 

step函数需要在优化器的 `step()` 函数之后调用，其根据传入的 metrics 调整optimizer中的学习率，调整后的学习率将会在下一个 ``step`` 时生效。

参数：
    metrics （Tensor|numpy.ndarray|float）-用来判断是否需要降低学习率。如果 loss 连续 patience 个 ``steps`` 没有下降， 将会降低学习率。可以是Tensor或者numpy.array，但是shape必须为[1] 。
  - **epoch** （int，可选）- 指定具体的epoch数。默认值None，此时将会从-1自动累加 ``epoch`` 数。

返回：
    无

**代码示例**:

    参照上述示例代码。
