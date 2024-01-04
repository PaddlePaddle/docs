.. _cn_api_paddle_callbacks_WandbCallback:

WandbCallback
-------------------------------

.. py:class:: paddle.callbacks.WandbCallback(project=None, entity=None, name=None, dir=None, mode=None, job_type=None, **kwargs)

使用 `Weights and Biases <https://docs.wandb.ai>`_ 跟踪您的训练和系统指标。

**安装和设置**
使用 pip 安装并登录您的 W&B 账户:

.. code-block:: bash

    pip install wandb
    wandb login

参数
::::::::::::

    - **project**(str, 可选) - 项目的名称。默认值为 uncategorized
    - **entity**(str, 可选) - 创建运行的团队/用户的名称。默认值为已登录的用户
    - **name**(str, 可选) - 运行的名称。默认值为 wandb 随机生成
    - **dir**(str, 可选) - 存储所有元数据的目录。默认值为 `wandb`
    - **mode**(str, 可选) - 可以是 "online"、"offline" 或 "disabled"。默认值为 "online".
    - **job_type**(str, 可选) -  运行类型，用于将运行分组在一起。默认值为 None


代码示例
::::::::::::

    COPY-FROM: paddle.callbacks.WandbCallback
