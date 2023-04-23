.. _cn_api_incubate_ModelAverage:

ModelAverage
-------------------------------

.. py:class:: paddle.incubate.ModelAverage(average_window_rate, parameters=None, min_average_window=10000, max_average_window=10000, name=None)

ModelAverage 优化器，在训练过程中累积特定连续的历史 ``Parameters``，累积的历史范围可以用传入的 ``average_window`` 参数来控制，在预测时使用平均后的 ``Parameters``，通常可以提高预测的精度。

在滑动窗口中累积 ``Parameters`` 的平均值，将结果将保存在临时变量中，通过调用 ``apply()`` 方法可应用于当前模型的 ``Parameters``，使用 ``restore()`` 方法恢复当前模型 ``Parameters`` 的值。

计算平均值的窗口大小由 ``average_window_rate`` ， ``min_average_window`` ， ``max_average_window`` 以及当前 ``Parameters`` 更新次数(num_updates)共同决定。

累积次数（num_accumulates）大于特定窗口阈值 (average_window) 时，将累积的 ``Parameters`` 临时变量置为 0.0。

``num_accumulates`` 表示当前累积的次数，可以抽象理解为累积窗口的长度；窗口长度至少要达到 ``min_average_window`` 参数设定的长度，并且不能超过 ``max_average_window`` 参数或者 ``num_updates`` * ``average_window_rate`` 规定的长度，否则为 0；而其中 ``num_updates`` 表示当前 ``Parameters`` 更新的次数，``average_window_rate`` 是一个计算窗口长度的系数。

参数
:::::::::
    - **average_window_rate** (float) – 相对于 ``Parameters`` 更新次数的窗口长度计算比率。
    - **parameters** (list，可选) - 为了最小化 ``loss`` 需要更新的 Tensor 列表。动态图模式下该参数是必需的；静态图模型下该参数的默认值为 None，此时所有参数都会被更新。
    - **min_average_window** (int，可选) – 平均值计算窗口长度的最小值，默认值为 10000。
    - **max_average_window** (int，可选) – 平均值计算窗口长度的最大值，默认值为 10000。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

代码示例
:::::::::
COPY-FROM: paddle.incubate.ModelAverage


方法
:::::::::

minimize(loss, startup_program=None, parameters=None, no_grad_set=None)
'''''''''

通过更新 ``Parameters`` 来最小化 ``loss`` 的方法。

**参数**

    - **loss** (Tensor) – 一个包含需要最小化的损失值变量的 Tensor。
    - **startup_program** (Program，可选) - 用于初始化 ``Parameters`` 中参数的 ``Program``，默认值为 None，此时将使用 ``default_startup_program``。
    - **parameters** (list，可选) – 待更新的 ``Parameters`` 或者 ``Parameter.name`` 组成的列表，默认值为 None，此时将更新所有的 ``Parameters``。
    - **no_grad_set** (set，可选) – 不需要更新的 ``Parameters`` 或者 ``Parameter.name`` 组成的集合，默认值为 None。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

**返回**

tuple(optimize_ops, params_grads)，其中 optimize_ops 为参数优化 OP 列表；param_grads 为由 (param, param_grad) 组成的列表，其中 param 和 param_grad 分别为参数和参数的梯度。该返回值可以加入到 ``Executor.run()`` 接口的 ``fetch_list`` 参数中，若加入，则会重写 ``use_prune`` 参数为 True，并根据 ``feed`` 和 ``fetch_list`` 进行剪枝，详见 ``Executor`` 的文档。

**代码示例**

COPY-FROM: paddle.incubate.ModelAverage.minimize


step()
'''''''''

执行一次优化器并进行参数更新。

**返回**

None。

**代码示例**

COPY-FROM: paddle.incubate.ModelAverage.step


apply(executor=None, need_restore=True)
'''''''''

将累积 ``Parameters`` 的平均值应用于当前网络的 ``Parameters``。

**参数**

    - **executor** (Executor) – 静态图模式下当前网络的执行器；动态图模式下默认值为 None。
    - **need_restore** (bool) - 恢复标志变量；设为 True 时，执行完成后会将网络的 ``Parameters``恢复为网络默认的值，设为 False 将不会恢复。默认值为 True。

**代码示例**

COPY-FROM: paddle.incubate.ModelAverage.apply


restore(executor=None)
'''''''''

恢复当前网络的 ``Parameters`` 值。

**参数**

    - **executor** (Executor) – 静态图模式下当前网络的执行器；动态图模式下默认值为 None。

**代码示例**

COPY-FROM: paddle.incubate.ModelAverage.restore
