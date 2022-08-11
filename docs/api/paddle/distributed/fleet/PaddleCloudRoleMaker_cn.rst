.. _cn_api_distributed_fleet_PaddleCloudRoleMaker:

PaddleCloudRoleMaker
-------------------------------

.. py:class:: paddle.distributed.fleet.PaddleCloudRoleMaker

PaddleCloudRoleMaker 是基于从环境变量中获取分布式相关信息进行分布式配置初始化的接口。
它会自动根据用户在环境变量中的配置进行分布式训练环境初始化，目前 PaddleCloudRoleMaker 支持 ParameterServer 分布式训练及 Collective 分布式训练两种模式的初始化。


代码示例
::::::::::::

.. code-block:: python

    import os
    import paddle.distributed.fleet as fleet

    os.environ["PADDLE_PSERVER_NUMS"] = "2"
    os.environ["PADDLE_TRAINERS_NUM"] = "2"

    os.environ["POD_IP"] = "127.0.0.1"
    os.environ["PADDLE_PORT"] = "36001"
    os.environ["TRAINING_ROLE"] = "PSERVER"
    os.environ["PADDLE_PSERVERS_IP_PORT_LIST"] = \
        "127.0.0.1:36001,127.0.0.2:36001"

    os.environ["PADDLE_TRAINER_ID"] = "0"

    fleet.PaddleCloudRoleMaker(is_collective=False)

方法
::::::::::::

to_string()
'''''''''

将当前环境变量以字符串的形式输出

**返回**

string


**代码示例**

.. code-block:: python

    import paddle.distributed.fleet as fleet
    role = fleet.PaddleCloudRoleMaker(is_collective=False)
    role.to_string()
