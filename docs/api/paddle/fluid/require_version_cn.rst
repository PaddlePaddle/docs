.. _cn_api_fluid_require_version:

require_version
-------------------------------

.. py:function:: paddle.fluid.require_version(min_version, max_version=None)



该接口用于检查已安装的飞桨版本是否介于[``min_version``, ``max_version``]之间（包含 ``min_version`` 和 ``max_version`` ），如果已安装的版本低于 ``min_version`` 或者高于 ``max_version``，将会抛出异常。该接口无返回值。

参数
::::::::::::

    - **min_version** (str) - 指定所需要的最低版本（如‘1.4.0’）
    - **max_version** (str，可选) – 指定可接受的最高版本（如‘1.7.0’），默认值 None，表示任意大于等于 ``min_version`` 的版本都可以接受。

返回
::::::::::::
无

抛出异常
::::::::::::


  - ``TypeError`` – ``min_version`` 的类型不是 str。
  - ``TypeError`` – ``max_version`` 的类型不是 str 或 type(None)。
  - ``ValueError`` – ``min_version`` 的值不是正常的版本号格式。
  - ``ValueError`` – ``max_version`` 的值不是正常的版本号格式或 None。
  - ``Exception`` – 已安装的版本低于 ``min_version`` 或者高于 ``max_version`` 。


代码示例
::::::::::::

COPY-FROM: paddle.fluid.require_version
