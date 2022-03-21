.. _cn_api_paddle_version_show:

show
-------------------------------

.. py:function:: paddle.version.show(debug=True)

如果paddle wheel包是正式发行版本，则打印版本号。否则，获取paddle wheel包编译时对应的commit id。
另外，打印paddle wheel包使用的CUDA和cuDNN的版本信息。


返回
:::::::::

如果paddle wheel包不是正式发行版本，则输出wheel包编译时对应的commit id号。否则，输出如下信息：

    - full_version - paddle wheel包的版本号。
    - major - paddle wheel包版本号的major信息。
    - minor - paddle wheel包版本号的minor信息。
    - patch - paddle wheel包版本号的patch信息。
    - rc - 是否是rc版本。
    - cuda - 若paddle wheel包为GPU版本，则返回paddle wheel包编译时使用的CUDA的版本信息；若paddle wheel包为CPU版本，则返回 ``False`` 。
    - cudnn - 若paddle wheel包为GPU版本，则返回paddle wheel包编译时使用的cuDNN的版本信息；若paddle wheel包为CPU版本，则返回 ``False`` 。

代码示例
::::::::::
COPY-FROM: paddle.version.show
