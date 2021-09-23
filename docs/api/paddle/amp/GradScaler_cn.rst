.. _cn_api_amp_GradScaler:

GradScaler
-------------------------------

.. py:class:: paddle.amp.GradScaler(enable=True, init_loss_scaling=32768.0, incr_ratio=2.0, decr_ratio=0.5, incr_every_n_steps=1000, decr_every_n_nan_or_inf=2, use_dynamic_loss_scaling=True)



GradScaler用于动态图模式下的"自动混合精度"的训练。它控制loss的缩放比例，有助于避免浮点数溢出的问题。这个类具有 ``scale()``、 ``unscale_()``、 ``step()``、 ``update()``、 ``minimize()``和参数的``get()/set()``等方法。

``scale()`` 用于让loss乘上一个缩放的比例。
``unscale_()`` 用于让loss除去一个缩放的比例。
``step()`` 与 ``optimizer.step()`` 类似，执行参数的更新，不更新缩放比例loss_scaling。
``update()`` 更新缩放比例。
``minimize()`` 与 ``optimizer.minimize()`` 类似，执行参数的更新，同时更新缩放比例loss_scaling，等效与``step()``+``update()``。

通常，GradScaler和 ``paddle.amp.auto_cast`` 一起使用，来实现动态图模式下的"自动混合精度"。


参数
:::::::::
    - **enable** (bool, 可选) - 是否使用loss scaling。默认值为True。
    - **init_loss_scaling** (float, 可选) - 初始loss scaling因子。默认值为32768.0。
    - **incr_ratio** (float, 可选) - 增大loss scaling时使用的乘数。默认值为2.0。
    - **decr_ratio** (float, 可选) - 减小loss scaling时使用的小于1的乘数。默认值为0.5。
    - **incr_every_n_steps** (int, 可选) - 连续n个steps的梯度都是有限值时，增加loss scaling。默认值为1000。
    - **decr_every_n_nan_or_inf** (int, 可选) - 累计出现n个steps的梯度为nan或者inf时，减小loss scaling。默认值为2。
    - **use_dynamic_loss_scaling** (bool, 可选) - 是否使用动态的loss scaling。如果不使用，则使用固定的loss scaling；如果使用，则会动态更新loss scaling。默认值为True。

返回
:::::::::
    一个GradScaler对象。


代码示例
:::::::::

.. code-block:: python

    import paddle

    model = paddle.nn.Conv2D(3, 2, 3, bias_attr=True)
    optimizer = paddle.optimizer.SGD(learning_rate=0.01, parameters=model.parameters())
    scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
    data = paddle.rand([10, 3, 32, 32])

    with paddle.amp.auto_cast():
        conv = model(data)
        loss = paddle.mean(conv)

    scaled = scaler.scale(loss)  # scale the loss 
    scaled.backward()            # do backward
    scaler.minimize(optimizer, scaled)  # update parameters
    optimizer.clear_grad()


scale(var)
'''''''''

将Tensor乘上缩放因子，返回缩放后的输出。
如果这个 :class:`GradScaler` 的实例不使用loss scaling，则返回的输出将保持不变。

**参数**
    - **var** (Tensor) - 需要进行缩放的Tensor。

**返回：**缩放后的Tensor或者原Tensor。

**代码示例：**

.. code-block:: python

    import paddle

    model = paddle.nn.Conv2D(3, 2, 3, bias_attr=True)
    optimizer = paddle.optimizer.SGD(learning_rate=0.01, parameters=model.parameters())
    scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
    data = paddle.rand([10, 3, 32, 32])

    with paddle.amp.auto_cast():
        conv = model(data)
        loss = paddle.mean(conv)

    scaled = scaler.scale(loss)  # scale the loss 
    scaled.backward()            # do backward
    scaler.minimize(optimizer, scaled)  # update parameters
    optimizer.clear_grad()

minimize(optimizer, *args, **kwargs)
'''''''''

这个函数与 ``optimizer.minimize()`` 类似，用于执行参数更新。
如果参数缩放后的梯度包含NAN或者INF，则跳过参数更新。否则，首先让缩放过梯度的参数取消缩放，然后更新参数。
最终，更新loss scaling的比例。

**参数：**
    - **optimizer** (Optimizer) - 用于更新参数的优化器。
    - **args** - 参数，将会被传递给 ``optimizer.minimize()`` 。
    - **kwargs** - 关键词参数，将会被传递给 ``optimizer.minimize()`` 。

**代码示例：**

.. code-block:: python

    import paddle

    model = paddle.nn.Conv2D(3, 2, 3, bias_attr=True)
    optimizer = paddle.optimizer.SGD(learning_rate=0.01, parameters=model.parameters())
    scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
    data = paddle.rand([10, 3, 32, 32])
    
    with paddle.amp.auto_cast():
        conv = model(data)
        loss = paddle.mean(conv)

    scaled = scaler.scale(loss)  # scale the loss 
    scaled.backward()            # do backward
    scaler.minimize(optimizer, scaled)  # update parameters
    optimizer.clear_grad()

step(optimizer)
'''''''''

这个函数与 ``optimizer.step()`` 类似，用于执行参数更新。
如果参数缩放后的梯度包含NAN或者INF，则跳过参数更新。否则，首先让缩放过梯度的参数取消缩放，然后更新参数。
该函数与 ``update()`` 函数一起使用，效果等同于 ``minimize()``。

**参数：**
    - **optimizer** (Optimizer) - 用于更新参数的优化器。

**代码示例：**

.. code-block:: python

    import paddle

    model = paddle.nn.Conv2D(3, 2, 3, bias_attr=True)
    optimizer = paddle.optimizer.SGD(learning_rate=0.01, parameters=model.parameters())
    scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
    data = paddle.rand([10, 3, 32, 32])
    with paddle.amp.auto_cast():
        conv = model(data)
        loss = paddle.mean(conv)
    scaled = scaler.scale(loss)  # scale the loss 
    scaled.backward()            # do backward
    scaler.step(optimizer)       # update parameters
    scaler.update()              # update the loss scaling ratio
    optimizer.clear_grad()

.. py:function:: update()

更新缩放比例。

代码示例：

.. code-block:: python

    import paddle

    model = paddle.nn.Conv2D(3, 2, 3, bias_attr=True)
    optimizer = paddle.optimizer.SGD(learning_rate=0.01, parameters=model.parameters())
    scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
    data = paddle.rand([10, 3, 32, 32])
    with paddle.amp.auto_cast():
        conv = model(data)
        loss = paddle.mean(conv)
    scaled = scaler.scale(loss)  # scale the loss 
    scaled.backward()            # do backward
    scaler.step(optimizer)       # update parameters
    scaler.update()              # update the loss scaling ratio
    optimizer.clear_grad()

.. py:function:: unscale_(optimizer)

将参数的梯度除去缩放比例。
如果在 ``step()`` 调用前调用 ``unscale_()``，则 ``step()`` 不会重复调用 ``unscale()``，否则 ``step()`` 将先执行 ``unscale_()`` 再做参数更新。
``minimize()`` 用法同上。

参数：
    - **optimizer** (Optimizer) - 用于更新参数的优化器。

代码示例：

.. code-block:: python

    import paddle

    model = paddle.nn.Conv2D(3, 2, 3, bias_attr=True)
    optimizer = paddle.optimizer.SGD(learning_rate=0.01, parameters=model.parameters())
    scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
    data = paddle.rand([10, 3, 32, 32])
    with paddle.amp.auto_cast():
        conv = model(data)
        loss = paddle.mean(conv)
    scaled = scaler.scale(loss)  # scale the loss 
    scaled.backward()            # do backward
    scaler.unscale_(optimizer)    # unscale the parameter
    scaler.step(optimizer)
    scaler.update()  
    optimizer.clear_grad() 

.. py:function:: is_enable()

判断是否开启loss scaling策略。

返回：bool，采用loss scaling策略返回True，否则返回False。

代码示例：

.. code-block:: python

    import paddle
    scaler = paddle.amp.GradScaler(enable=True,
                                   init_loss_scaling=1024,
                                   incr_ratio=2.0,
                                   decr_ratio=0.5,
                                   incr_every_n_steps=1000,
                                   decr_every_n_nan_or_inf=2,
                                   use_dynamic_loss_scaling=True)
    enable = scaler.is_enable()
    print(enable) # True

.. py:function:: is_use_dynamic_loss_scaling()

判断是否动态调节loss scaling的缩放比例。

返回：bool，动态调节loss scaling缩放比例返回True，否则返回False。

代码示例：

.. code-block:: python

    import paddle
    scaler = paddle.amp.GradScaler(enable=True,
                                   init_loss_scaling=1024,
                                   incr_ratio=2.0,
                                   decr_ratio=0.5,
                                   incr_every_n_steps=1000,
                                   decr_every_n_nan_or_inf=2,
                                   use_dynamic_loss_scaling=True)
    use_dynamic_loss_scaling = scaler.is_use_dynamic_loss_scaling()
    print(use_dynamic_loss_scaling) # True

.. py:function:: get_init_loss_scaling()

返回初始化的loss scaling缩放比例。

返回：float，初始化的loss scaling缩放比例。

代码示例：

.. code-block:: python

    import paddle
    scaler = paddle.amp.GradScaler(enable=True,
                                   init_loss_scaling=1024,
                                   incr_ratio=2.0,
                                   decr_ratio=0.5,
                                   incr_every_n_steps=1000,
                                   decr_every_n_nan_or_inf=2,
                                   use_dynamic_loss_scaling=True)
    init_loss_scaling = scaler.get_init_loss_scaling()
    print(init_loss_scaling) # 1024

.. py:function:: set_init_loss_scaling(new_init_loss_scaling)

利用输入的new_init_loss_scaling对初始缩放比例参数init_loss_scaling重新赋值。

参数：
    - **new_init_loss_scaling** (float) - 用于更新缩放比例的参数。

代码示例：

.. code-block:: python

    import paddle
    scaler = paddle.amp.GradScaler(enable=True,
                                   init_loss_scaling=1024,
                                   incr_ratio=2.0,
                                   decr_ratio=0.5,
                                   incr_every_n_steps=1000,
                                   decr_every_n_nan_or_inf=2,
                                   use_dynamic_loss_scaling=True)
    print(scaler.get_init_loss_scaling()) # 1024
    new_init_loss_scaling = 1000
    scaler.set_init_loss_scaling(new_init_loss_scaling)
    print(scaler.get_init_loss_scaling()) # 1000

.. py:function:: get_incr_ratio()

返回增大loss scaling时使用的乘数。

返回：float，增大loss scaling时使用的乘数。

代码示例：

.. code-block:: python

    import paddle
    scaler = paddle.amp.GradScaler(enable=True,
                                   init_loss_scaling=1024,
                                   incr_ratio=2.0,
                                   decr_ratio=0.5,
                                   incr_every_n_steps=1000,
                                   decr_every_n_nan_or_inf=2,
                                   use_dynamic_loss_scaling=True)
    incr_ratio = scaler.get_incr_ratio()
    print(incr_ratio) # 2.0

.. py:function:: set_incr_ratio(new_incr_ratio)

利用输入的new_incr_ratio对增大loss scaling时使用的乘数重新赋值。

参数：
    - **new_incr_ratio** (float) - 用于更新增大loss scaling时使用的乘数，该值需>1.0。

代码示例：

.. code-block:: python

    import paddle
    scaler = paddle.amp.GradScaler(enable=True,
                                   init_loss_scaling=1024,
                                   incr_ratio=2.0,
                                   decr_ratio=0.5,
                                   incr_every_n_steps=1000,
                                   decr_every_n_nan_or_inf=2,
                                   use_dynamic_loss_scaling=True)
    print(scaler.get_incr_ratio()) # 2.0
    new_incr_ratio = 3.0
    scaler.set_incr_ratio(new_incr_ratio)
    print(scaler.get_incr_ratio()) # 3.0

.. py:function:: get_decr_ratio()

返回缩小loss scaling时使用的乘数。

返回：float，缩小loss scaling时使用的乘数。

代码示例：

.. code-block:: python

    import paddle
    scaler = paddle.amp.GradScaler(enable=True,
                                   init_loss_scaling=1024,
                                   incr_ratio=2.0,
                                   decr_ratio=0.5,
                                   incr_every_n_steps=1000,
                                   decr_every_n_nan_or_inf=2,
                                   use_dynamic_loss_scaling=True)
    decr_ratio = scaler.get_decr_ratio()
    print(decr_ratio) # 0.5

.. py:function:: set_decr_ratio(new_decr_ratio)

利用输入的new_decr_ratio对缩小loss scaling时使用的乘数重新赋值。

参数：
    - **new_decr_ratio** (float) - 用于更新缩小loss scaling时使用的乘数，该值需<1.0。

代码示例：

.. code-block:: python

    import paddle
    scaler = paddle.amp.GradScaler(enable=True,
                                   init_loss_scaling=1024,
                                   incr_ratio=2.0,
                                   decr_ratio=0.5,
                                   incr_every_n_steps=1000,
                                   decr_every_n_nan_or_inf=2,
                                   use_dynamic_loss_scaling=True)
    print(scaler.get_decr_ratio()) # 0.5
    new_decr_ratio = 0.1
    scaler.set_decr_ratio(new_decr_ratio)
    print(scaler.get_decr_ratio()) # 0.1

.. py:function:: get_incr_every_n_steps()

连续n个steps的梯度都是有限值时，增加loss scaling，返回对应的值n。

返回：int，参数incr_every_n_steps。

代码示例：

.. code-block:: python

    import paddle
    scaler = paddle.amp.GradScaler(enable=True,
                                   init_loss_scaling=1024,
                                   incr_ratio=2.0,
                                   decr_ratio=0.5,
                                   incr_every_n_steps=1000,
                                   decr_every_n_nan_or_inf=2,
                                   use_dynamic_loss_scaling=True)
    incr_every_n_steps = scaler.get_incr_every_n_steps()
    print(incr_every_n_steps) # 1000

.. py:function:: set_incr_every_n_steps(new_incr_every_n_steps)

利用输入的new_incr_every_n_steps对参数incr_every_n_steps重新赋值。

参数：
    - **new_incr_every_n_steps** (int) - 用于更新参数incr_every_n_steps。

代码示例：

.. code-block:: python

    import paddle
    scaler = paddle.amp.GradScaler(enable=True,
                                   init_loss_scaling=1024,
                                   incr_ratio=2.0,
                                   decr_ratio=0.5,
                                   incr_every_n_steps=1000,
                                   decr_every_n_nan_or_inf=2,
                                   use_dynamic_loss_scaling=True)
    print(scaler.get_incr_every_n_steps()) # 1000
    new_incr_every_n_steps = 2000
    scaler.set_incr_every_n_steps(new_incr_every_n_steps)
    print(scaler.get_incr_every_n_steps()) # 2000

.. py:function:: get_decr_every_n_nan_or_inf()

累计出现n个steps的梯度为nan或者inf时，减小loss scaling，返回对应的值n。

返回：int，参数decr_every_n_nan_or_inf。

代码示例：

.. code-block:: python

    import paddle
    scaler = paddle.amp.GradScaler(enable=True,
                                   init_loss_scaling=1024,
                                   incr_ratio=2.0,
                                   decr_ratio=0.5,
                                   incr_every_n_steps=1000,
                                   decr_every_n_nan_or_inf=2,
                                   use_dynamic_loss_scaling=True)
    decr_every_n_nan_or_inf = scaler.get_decr_every_n_nan_or_inf()
    print(decr_every_n_nan_or_inf) # 2

.. py:function:: set_decr_every_n_nan_or_inf(new_decr_every_n_nan_or_inf)

利用输入的new_decr_every_n_nan_or_inf对参数decr_every_n_nan_or_inf重新赋值。

参数：
    - **new_decr_every_n_nan_or_inf** (int) - 用于更新参数decr_every_n_nan_or_inf。

代码示例：

.. code-block:: python

    import paddle
    scaler = paddle.amp.GradScaler(enable=True,
                                   init_loss_scaling=1024,
                                   incr_ratio=2.0,
                                   decr_ratio=0.5,
                                   incr_every_n_steps=1000,
                                   decr_every_n_nan_or_inf=2,
                                   use_dynamic_loss_scaling=True)
    print(scaler.get_decr_every_n_nan_or_inf()) # 2
    new_decr_every_n_nan_or_inf = 3
    scaler.set_decr_every_n_nan_or_inf(new_decr_every_n_nan_or_inf)
    print(scaler.get_decr_every_n_nan_or_inf()) # 3

.. py:function:: state_dict()

以字典的形式存储GradScaler对象的状态参数，如果该对象的enable为False，则返回一个空的字典。

返回：dict，字典存储的参数包括：init_loss_scaling(float):初始loss scaling因子、incr_ratio(float):增大loss scaling时使用的乘数、decr_ratio(float):减小loss scaling时使用的小于1的乘数、incr_every_n_steps(int):连续n个steps的梯度都是有限值时，增加loss scaling、decr_every_n_nan_or_inf(int):累计出现n个steps的梯度为nan或者inf时，减小loss scaling。

代码示例：

.. code-block:: python

    import paddle

    scaler = paddle.amp.GradScaler(enable=True,
                                   init_loss_scaling=1024,
                                   incr_ratio=2.0,
                                   decr_ratio=0.5,
                                   incr_every_n_steps=1000,
                                   decr_every_n_nan_or_inf=2,
                                   use_dynamic_loss_scaling=True)
    scaler_state = scaler.state_dict()

.. py:function:: load_state_dict(state_dict)

利用输入的state_dict设置或更新GradScaler对象的属性参数。

参数：
    - **state_dict** (dict) - 用于设置或更新GradScaler对象的属性参数，dict需要是``GradScaler.state_dict()``的返回值。

代码示例：

.. code-block:: python

    import paddle

    scaler = paddle.amp.GradScaler(enable=True,
                                   init_loss_scaling=1024,
                                   incr_ratio=2.0,
                                   decr_ratio=0.5,
                                   incr_every_n_steps=1000,
                                   decr_every_n_nan_or_inf=2,
                                   use_dynamic_loss_scaling=True)
    scaler_state = scaler.state_dict()
    scaler.load_state_dict(scaler_state)
