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