.. _cn_api_optimizer_MultiStepLR:

MultiStepLR
-----------------------------------

.. py:class:: paddle.optimizer.MultiStepLR(learning_rate, milestones, gamma=0.1, last_epoch=-1, verbose=False)

该接口提供一种学习率按指定轮数衰减的功能。

衰减过程可以参考以下代码：

.. code-block:: python

      learning_rate = 0.5
      milestones = [30, 50]
      gamma = 0.1
      if epoch < 30:
          learning_rate = 0.5
      elif epoch < 50:
          learning_rate = 0.05  # 0.5 * 0.1
      else:
          learning_rate = 0.005 # 0.05 * 0.1

参数
:::::::::
    - **learning_rate** （float|int）：初始学习率，可以是Python的float或int。
    - **milestones** ：（list）：轮数下标列表。必须递增。
    - **gamma** （float）：衰减率。
    - **last_epoch** （int）：上一轮的下标。默认为 `-1` 。
    - **verbose** （bool）：如果是 `True` ，则在每一轮更新时在标准输出 `stdout` 输出一条信息。


返回
:::::::::
    无

代码示例
:::::::::

.. code-block:: python


