.. _cn_api_amp_GradScaler:

GradScaler
-------------------------------

.. py:class:: paddle.amp.GradScaler(enable=True, init_loss_scaling=65536.0, incr_ratio=2.0, decr_ratio=0.5, incr_every_n_steps=2000, decr_every_n_nan_or_inf=1, use_dynamic_loss_scaling=True)



GradScaler 用于动态图模式下的"自动混合精度"的训练。它控制 loss 的缩放比例，有助于避免浮点数溢出的问题。这个类具有 ``scale()``、 ``unscale_()``、 ``step()``、 ``update()``、 ``minimize()``和参数的``get()/set()``等方法。

``scale()`` 用于让 loss 乘上一个缩放的比例。
``unscale_()`` 用于让 loss 除去一个缩放的比例。
``step()`` 与 ``optimizer.step()`` 类似，执行参数的更新，不更新缩放比例 loss_scaling。
``update()`` 更新缩放比例。
``minimize()`` 与 ``optimizer.minimize()`` 类似，执行参数的更新，同时更新缩放比例 loss_scaling，等效与``step()``+``update()``。

通常，GradScaler 和 ``paddle.amp.auto_cast`` 一起使用，来实现动态图模式下的"自动混合精度"。


参数
:::::::::
    - **enable** (bool，可选) - 是否使用 loss scaling。默认值为 True。
    - **init_loss_scaling** (float，可选) - 初始 loss scaling 因子。默认值为 65536.0。
    - **incr_ratio** (float，可选) - 增大 loss scaling 时使用的乘数。默认值为 2.0。
    - **decr_ratio** (float，可选) - 减小 loss scaling 时使用的小于 1 的乘数。默认值为 0.5。
    - **incr_every_n_steps** (int，可选) - 连续 n 个 steps 的梯度都是有限值时，增加 loss scaling。默认值为 2000。
    - **decr_every_n_nan_or_inf** (int，可选) - 累计出现 n 个 steps 的梯度为 nan 或者 inf 时，减小 loss scaling。默认值为 1。
    - **use_dynamic_loss_scaling** (bool，可选) - 是否使用动态的 loss scaling。如果不使用，则使用固定的 loss scaling；如果使用，则会动态更新 loss scaling。默认值为 True。

返回
:::::::::
    一个 GradScaler 对象。


代码示例
:::::::::

COPY-FROM: paddle.amp.GradScaler


scale(var)
'''''''''

将 Tensor 乘上缩放因子，返回缩放后的输出。
如果这个 :class:`GradScaler` 的实例不使用 loss scaling，则返回的输出将保持不变。

**参数**

- **var** (Tensor) - 需要进行缩放的 Tensor。

**返回**

缩放后的 Tensor 或者原 Tensor。

**代码示例**

COPY-FROM: paddle.amp.GradScaler.scale

minimize(optimizer, *args, **kwargs)
'''''''''

这个函数与 ``optimizer.minimize()`` 类似，用于执行参数更新。
如果参数缩放后的梯度包含 NAN 或者 INF，则跳过参数更新。否则，首先让缩放过梯度的参数取消缩放，然后更新参数。
最终，更新 loss scaling 的比例。

**参数**

    - **optimizer** (Optimizer) - 用于更新参数的优化器。
    - **args** - 参数，将会被传递给 ``optimizer.minimize()`` 。
    - **kwargs** - 关键词参数，将会被传递给 ``optimizer.minimize()`` 。

**代码示例**

COPY-FROM: paddle.amp.GradScaler.minimize

step(optimizer)
'''''''''

这个函数与 ``optimizer.step()`` 类似，用于执行参数更新。
如果参数缩放后的梯度包含 NAN 或者 INF，则跳过参数更新。否则，首先让缩放过梯度的参数取消缩放，然后更新参数。
该函数与 ``update()`` 函数一起使用，效果等同于 ``minimize()``。

**参数**

- **optimizer** (Optimizer) - 用于更新参数的优化器。

**代码示例**

COPY-FROM: paddle.amp.GradScaler.step

update()
'''''''''

更新缩放比例。

**代码示例**

COPY-FROM: paddle.amp.GradScaler.update

unscale_(optimizer)
'''''''''

将参数的梯度除去缩放比例。
如果在 ``step()`` 调用前调用 ``unscale_()``，则 ``step()`` 不会重复调用 ``unscale()``，否则 ``step()`` 将先执行 ``unscale_()`` 再做参数更新。
``minimize()`` 用法同上。

**参数**
    - **optimizer** (Optimizer) - 用于更新参数的优化器。

**代码示例**

COPY-FROM: paddle.amp.GradScaler.unscale_

is_enable()
'''''''''

判断是否开启 loss scaling 策略。

**返回**

bool，采用 loss scaling 策略返回 True，否则返回 False。

**代码示例**

COPY-FROM: paddle.amp.GradScaler.is_enable

is_use_dynamic_loss_scaling()
'''''''''

判断是否动态调节 loss scaling 的缩放比例。

**返回**

bool，动态调节 loss scaling 缩放比例返回 True，否则返回 False。

**代码示例**

COPY-FROM: paddle.amp.GradScaler.is_use_dynamic_loss_scaling

get_init_loss_scaling()
'''''''''

返回初始化的 loss scaling 缩放比例。

**返回**

float，初始化的 loss scaling 缩放比例。

**代码示例**

COPY-FROM: paddle.amp.GradScaler.get_init_loss_scaling

set_init_loss_scaling(new_init_loss_scaling)
'''''''''

利用输入的 new_init_loss_scaling 对初始缩放比例参数 init_loss_scaling 重新赋值。

**参数**

- **new_init_loss_scaling** (float) - 用于更新缩放比例的参数。

**代码示例**

COPY-FROM: paddle.amp.GradScaler.set_init_loss_scaling

get_incr_ratio()
'''''''''

返回增大 loss scaling 时使用的乘数。

**返回**

float，增大 loss scaling 时使用的乘数。

**代码示例**

COPY-FROM: paddle.amp.GradScaler.get_incr_ratio

set_incr_ratio(new_incr_ratio)
'''''''''

利用输入的 new_incr_ratio 对增大 loss scaling 时使用的乘数重新赋值。

**参数**

- **new_incr_ratio** (float) - 用于更新增大 loss scaling 时使用的乘数，该值需>1.0。

**代码示例**

COPY-FROM: paddle.amp.GradScaler.set_incr_ratio

get_decr_ratio()
'''''''''

返回缩小 loss scaling 时使用的乘数。

**返回**

float，缩小 loss scaling 时使用的乘数。

**代码示例**

COPY-FROM: paddle.amp.GradScaler.get_decr_ratio

set_decr_ratio(new_decr_ratio)
'''''''''

利用输入的 new_decr_ratio 对缩小 loss scaling 时使用的乘数重新赋值。

**参数**

- **new_decr_ratio** (float) - 用于更新缩小 loss scaling 时使用的乘数，该值需<1.0。

**代码示例**

COPY-FROM: paddle.amp.GradScaler.set_decr_ratio

get_incr_every_n_steps()
'''''''''

连续 n 个 steps 的梯度都是有限值时，增加 loss scaling，返回对应的值 n。

**返回**

int，参数 incr_every_n_steps。

**代码示例**

COPY-FROM: paddle.amp.GradScaler.get_incr_every_n_steps

set_incr_every_n_steps(new_incr_every_n_steps)
'''''''''

利用输入的 new_incr_every_n_steps 对参数 incr_every_n_steps 重新赋值。

**参数**

- **new_incr_every_n_steps** (int) - 用于更新参数 incr_every_n_steps。

**代码示例**

COPY-FROM: paddle.amp.GradScaler.set_incr_every_n_steps

get_decr_every_n_nan_or_inf()
'''''''''

累计出现 n 个 steps 的梯度为 nan 或者 inf 时，减小 loss scaling，返回对应的值 n。

**返回**

int，参数 decr_every_n_nan_or_inf。

**代码示例**

COPY-FROM: paddle.amp.GradScaler.get_decr_every_n_nan_or_inf

set_decr_every_n_nan_or_inf(new_decr_every_n_nan_or_inf)
'''''''''

利用输入的 new_decr_every_n_nan_or_inf 对参数 decr_every_n_nan_or_inf 重新赋值。

**参数**

- **new_decr_every_n_nan_or_inf** (int) - 用于更新参数 decr_every_n_nan_or_inf。

**代码示例**

COPY-FROM: paddle.amp.GradScaler.set_decr_every_n_nan_or_inf

state_dict()
'''''''''

以字典的形式存储 GradScaler 对象的状态参数，如果该对象的 enable 为 False，则返回一个空的字典。

**返回**

dict，字典存储的参数包括：scale(tensor):loss scaling 因子、incr_ratio(float):增大 loss scaling 时使用的乘数、decr_ratio(float):减小 loss scaling 时使用的小于 1 的乘数、incr_every_n_steps(int):连续 n 个 steps 的梯度都是有限值时，增加 loss scaling、decr_every_n_nan_or_inf(int):累计出现 n 个 steps 的梯度为 nan 或者 inf 时，减小 loss scaling、incr_count(int):连续未跳过参数更新的次数、decr_count(int):连续跳过参数更新的次数、use_dynamic_loss_scaling(bool):是否使用动态 loss scaling 策略。

**代码示例**

COPY-FROM: paddle.amp.GradScaler.state_dict

load_state_dict(state_dict)
'''''''''

利用输入的 state_dict 设置或更新 GradScaler 对象的属性参数。

**参数**

- **state_dict** (dict) - 用于设置或更新 GradScaler 对象的属性参数，dict 需要是``GradScaler.state_dict()``的返回值。

**代码示例**

COPY-FROM: paddle.amp.GradScaler.load_state_dict
