.. _cn_api_paddle_callbacks_Callback:

Callback
-------------------------------

.. py:class:: paddle.callbacks.Callback()

 ``Callback`` 是一个基类，用于实现用户自定义的callback。

**代码示例**：

.. code-block:: python

    import paddle

    # build a simple model checkpoint callback
    class ModelCheckpoint(paddle.callbacks.Callback):
        def __init__(self, save_freq=1, save_dir=None):
            self.save_freq = save_freq
            self.save_dir = save_dir

        def on_epoch_end(self, epoch, logs=None):
            if self.model is not None and epoch % self.save_freq == 0:
                path = '{}/{}'.format(self.save_dir, epoch)
                print('save checkpoint at {}'.format(path))
                self.model.save(path)

方法
:::::::::

set_params(params)
'''''''''

设置参数，类型是dict，包含字段如下:

- 'batch_size': 整数，批大小
- ‘epochs’: 整数，总共epochs
- 'steps': 整数，一个epoch内的step数
- 'verbose': 整数，0，1，2，表示输出信息的模式，0是静默模式，1是进度条模式，2是每次打印一行。
- ‘metrics’: 字符串数组，评估指标的名字，包含’loss‘，以及paddle.metric.Metric获取的名字。

set_model(model)
'''''''''

设置paddle.Model实例。

on_train_begin(logs=None)
'''''''''

在训练的一开始调用。

参数：
    - **logs** (dict|None): 日志信息是dict或None.

on_train_end(logs=None)
'''''''''

在训练的结束调用。

参数：
    - **logs** (dict|None): 日志信息是dict或None. 通过paddle.Model传递的dict包含的字段有'loss', 评估指标metric的名字，以及'batch_size'。


on_eval_begin(logs=None)
'''''''''

在评估阶段的一开始调用。

参数：
    - **logs** (dict|None): 日志信息是dict或None. 通过paddle.Model传递的dict包含的字段有'steps'和'metrics'。'steps'是验证集的总共步长数, 'metrics'是一个list[str], 包含'loss'和所设置的paddle.metric.Metric的名字。

on_eval_end(logs=None)
'''''''''

在评估阶段的结束调用。

参数：
    - **logs** (dict|None): 日志信息是dict或None. 通过paddle.Model传递的dict包含的字段有'loss', 评估指标metric的名字，以及'batch_size'。


on_predict_begin(logs=None)
'''''''''

在推理阶段的一开始调用。

参数：
    - **logs** (dict|None): 日志信息是dict或None。


on_predict_end(logs=None)
'''''''''

在推理阶段的结束调用。

参数：
    - **logs** (dict|None): 日志信息是dict或None。


on_epoch_begin(epoch, logs=None)
'''''''''

在每个epoch的一开始调用。

参数：
    - **epoch** (int): epoch的索引。
    - **logs** (dict|None): 日志信息是None。

on_epoch_end(epoch, logs=None)
'''''''''

在每个epoch的结束调用。

参数：
    - **epoch** (int): epoch的索引。
    - **logs** (dict|None): 日志信息是dict或None. 通过paddle.Model传递的dict包含的字段有'loss', 评估指标metric的名字，以及'batch_size'。


on_train_batch_begin(step, logs=None)
'''''''''

在训练阶段每个batch的开始调用。

参数：
    - **step** (int): 训练步长或迭代次数。
    - **logs** (dict|None): 日志信息是dict或None. 通过paddle.Model传递的是None。


on_train_batch_end(step, logs=None)
'''''''''

在训练阶段每个batch的结束调用。

参数：
    - **step** (int): 训练步长或迭代次数。
    - **logs** (dict|None): 日志信息是dict或None. 通过paddle.Model传递的dict包含的字段有'loss', 评估指标metric的名字，以及当前'batch_size'。


on_eval_batch_begin(step, logs=None)
'''''''''

在评估阶段每个batch的开始调用。

参数：
    - **step** (int): 评估步长或迭代次数。
    - **logs** (dict|None): 日志信息是dict或None. 通过paddle.Model传递的是None。

on_eval_batch_end(step, logs=None)
'''''''''

在评估阶段每个batch的结束调用。

参数：
    - **step** (int): 训练步长或迭代次数。
    - **logs** (dict|None): 日志信息是dict或None. 通过paddle.Model传递的dict包含的字段有'loss', 评估指标metric的名字，以及当前'batch_size'。

on_predict_batch_begin(step, logs=None)
'''''''''

在推理阶段每个batch的开始调用。

参数：
    - **step** (int): 推理步长或迭代次数。
    - **logs** (dict|None): 日志信息是dict或None.

on_predict_batch_end(step, logs=None)
'''''''''

在推理阶段每个batch的结束调用。

参数：
    - **step** (int): 训练步长或迭代次数。
    - **logs** (dict|None): 日志信息是dict或None.
