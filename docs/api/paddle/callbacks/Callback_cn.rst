.. _cn_api_paddle_callbacks_Callback:

Callback
-------------------------------

.. py:class:: paddle.callbacks.Callback()

 ``Callback`` 是一个基类，用于实现用户自定义的 callback。如果想使用除 :ref:`EarlyStopping <_cn_api_paddle_callbacks_EarlyStopping>` 外的自定义策略终止训练，可以通过在自定义的 callback 类中设置 ``model.stop_training=True`` 来实现。

代码示例
::::::::::::


COPY-FROM: paddle.callbacks.Callback

方法
:::::::::

set_params(params)
'''''''''

设置参数，类型是 dict，包含字段如下：

- ‘batch_size’：整数，批大小。
- ‘epochs’：整数，总共 epochs。
- ‘steps’：整数，一个 epoch 内的 step 数。
- ‘verbose’：整数，0，1，2，表示输出信息的模式，0 是静默模式，1 是进度条模式，2 是每次打印一行。
- ‘metrics’：字符串数组，评估指标的名字，包含’loss‘，以及 paddle.metric.Metric 获取的名字。

set_model(model)
'''''''''

设置 paddle.Model 实例。

on_train_begin(logs=None)
'''''''''

在训练的一开始调用。

**参数**

    - **logs** (dict|None)：日志信息是 dict 或 None。

on_train_end(logs=None)
'''''''''

在训练的结束调用。

**参数**

    - **logs** (dict|None)：日志信息是 dict 或 None。通过 paddle.Model 传递的 dict 包含的字段有'loss'，评估指标 metric 的名字，以及'batch_size'。


on_eval_begin(logs=None)
'''''''''

在评估阶段的一开始调用。

**参数**

    - **logs** (dict|None)：日志信息是 dict 或 None。通过 paddle.Model 传递的 dict 包含的字段有'steps'和'metrics'。'steps'是验证集的总共步长数，'metrics'是一个 list[str]，包含'loss'和所设置的 paddle.metric.Metric 的名字。

on_eval_end(logs=None)
'''''''''

在评估阶段的结束调用。

**参数**

    - **logs** (dict|None)：日志信息是 dict 或 None。通过 paddle.Model 传递的 dict 包含的字段有'loss'，评估指标 metric 的名字，以及'batch_size'。


on_predict_begin(logs=None)
'''''''''

在推理阶段的一开始调用。

**参数**

    - **logs** (dict|None)：日志信息是 dict 或 None。


on_predict_end(logs=None)
'''''''''

在推理阶段的结束调用。

**参数**

    - **logs** (dict|None)：日志信息是 dict 或 None。


on_epoch_begin(epoch, logs=None)
'''''''''

在每个 epoch 的一开始调用。

**参数**

    - **epoch** (int): epoch 的索引。
    - **logs** (dict|None)：日志信息是 None。

on_epoch_end(epoch, logs=None)
'''''''''

在每个 epoch 的结束调用。

**参数**

    - **epoch** (int): epoch 的索引。
    - **logs** (dict|None)：日志信息是 dict 或 None。通过 paddle.Model 传递的 dict 包含的字段有'loss'，评估指标 metric 的名字，以及'batch_size'。


on_train_batch_begin(step, logs=None)
'''''''''

在训练阶段每个 batch 的开始调用。

**参数**

    - **step** (int)：训练步长或迭代次数。
    - **logs** (dict|None)：日志信息是 dict 或 None。通过 paddle.Model 传递的是 None。


on_train_batch_end(step, logs=None)
'''''''''

在训练阶段每个 batch 的结束调用。

**参数**

    - **step** (int)：训练步长或迭代次数。
    - **logs** (dict|None)：日志信息是 dict 或 None。通过 paddle.Model 传递的 dict 包含的字段有'loss'，评估指标 metric 的名字，以及当前'batch_size'。


on_eval_batch_begin(step, logs=None)
'''''''''

在评估阶段每个 batch 的开始调用。

**参数**

    - **step** (int)：评估步长或迭代次数。
    - **logs** (dict|None)：日志信息是 dict 或 None。通过 paddle.Model 传递的是 None。

on_eval_batch_end(step, logs=None)
'''''''''

在评估阶段每个 batch 的结束调用。

**参数**

    - **step** (int)：训练步长或迭代次数。
    - **logs** (dict|None)：日志信息是 dict 或 None。通过 paddle.Model 传递的 dict 包含的字段有'loss'，评估指标 metric 的名字，以及当前'batch_size'。

on_predict_batch_begin(step, logs=None)
'''''''''

在推理阶段每个 batch 的开始调用。

**参数**

    - **step** (int)：推理步长或迭代次数。
    - **logs** (dict|None)：日志信息是 dict 或 None。

on_predict_batch_end(step, logs=None)
'''''''''

在推理阶段每个 batch 的结束调用。

**参数**

    - **step** (int)：训练步长或迭代次数。
    - **logs** (dict|None)：日志信息是 dict 或 None。
