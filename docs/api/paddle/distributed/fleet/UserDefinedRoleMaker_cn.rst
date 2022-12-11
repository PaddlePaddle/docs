.. _cn_api_distributed_fleet_UserDefinedRoleMaker:

UserDefinedRoleMaker
-------------------------------

.. py:class:: paddle.distributed.fleet.UserDefinedRoleMaker

UserDefinedRoleMaker 是基于从用户自定义的参数中获取分布式相关信息进行分布式配置初始化的接口
它会自动根据用户的自定义配置进行分布式训练环境初始化，目前 UserDefinedRoleMaker 支持 ParameterServer 分布式训练及 Collective 分布式训练两种模式的初始化。


代码示例
::::::::::::

.. code-block:: python

    import paddle.distributed.fleet as fleet
    from paddle.distributed.fleet.base.role_maker import Role

    fleet.UserDefinedRoleMaker(
        current_id=0,
        role=Role.SERVER,
        worker_num=2,
        server_endpoints=["127.0.0.1:36011", "127.0.0.1:36012"])

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
    from paddle.distributed.fleet.base.role_maker import Role

    role = fleet.UserDefinedRoleMaker(
        current_id=0,
        role=Role.SERVER,
        worker_num=2,
        server_endpoints=["127.0.0.1:36011", "127.0.0.1:36012"])

    role.to_string()
