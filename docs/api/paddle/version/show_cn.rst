.. _cn_api_paddle_version_show:

show
-------------------------------

.. py:function:: paddle.version.show()

如果paddle包已经标记了特定版本，打印版本号。否则，获取paddle包对应的commit id。
打印paddle包使用的cuda和cudnn的版本信息。


返回：
:::::::::
如果paddle包没有被标记为特定版本，输出其对应的commit id。否则，输出如下信息：

    - full_version - paddle包的版本号。
    - major - paddle包版本号的major信息。
    - minor - paddle包版本号的minor信息。
    - patch - paddle包版本号的patch信息。
    - rc - 是否是rc版本。
    - cuda - paddle包使用的cuda版本。若安装的是CPU版本，则返回 ``False`` 。
    - cudnn - paddle包使用的cudnn版本。若安装的是CPU版本，则返回 ``False`` 。

代码示例：
::::::::::

.. code-block:: python

    import paddle

    # Case 1: paddle is tagged with 2.2.0
    paddle.version.show()
    # full_version: 2.2.0
    # major: 2
    # minor: 2
    # patch: 0
    # rc: 0
    # cuda: '10.2'
    # cudnn: '7.6.5'

    # Case 2: paddle is not tagged
    paddle.version.show()
    # commit: cfa357e984bfd2ffa16820e354020529df434f7d
    # cuda: '10.2'
    # cudnn: '7.6.5'

