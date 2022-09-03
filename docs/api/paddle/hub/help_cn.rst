.. _cn_api_paddle_hub_help:

help
-------------------------------

.. py:function:: paddle.hub.help(repo_dir, model, source='github', force_reload=False)


用于查看 repo 提供的功能/模型的文档。


参数
:::::::::

    - **repo_dir** (str) - repo 地址，支持 git 地址形式和 local 地址。git 地址由 repo 拥有者/repo 名字:repo 分支组成，实例：PaddlePaddle/PaddleClas:develop；local 地址为 repo 的本地路径。
    - **model** (str) - 模型的名字。
    - **source** (str，可选) - 指定 repo 托管的位置，支持 github、gitee 和 local，默认值：github。
    - **force_reload** (bool，可选) - 指定是否强制拉取，默认值: False。

返回
:::::::::

    ``str`` ，repo 提供的指定模型的文档。


代码示例
:::::::::

COPY-FROM: paddle.hub.help
