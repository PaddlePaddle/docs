.. _cn_api_paddle_version_show:

show
-------------------------------

.. py:function:: paddle.version.show()

如果 paddle wheel 包是正式发行版本，则打印版本号。否则，获取 paddle wheel 包编译时对应的 commit id。
另外，打印 paddle wheel 包使用的 CUDA 和 cuDNN 的版本信息。


返回
:::::::::

如果 paddle wheel 包不是正式发行版本，则输出 wheel 包编译时对应的 commit id 号。否则，输出如下信息：

    - full_version - paddle wheel 包的版本号。
    - major - paddle wheel 包版本号的 major 信息。
    - minor - paddle wheel 包版本号的 minor 信息。
    - patch - paddle wheel 包版本号的 patch 信息。
    - rc - 是否是 rc 版本。
    - cuda - 若 paddle wheel 包为 GPU 版本，则返回 paddle wheel 包编译时使用的 CUDA 的版本信息；若 paddle wheel 包为 CPU 版本，则返回 ``False`` 。
    - cudnn - 若 paddle wheel 包为 GPU 版本，则返回 paddle wheel 包编译时使用的 cuDNN 的版本信息；若 paddle wheel 包为 CPU 版本，则返回 ``False`` 。
    - tensorrt - 返回 paddle 安装包编译时使用的 TensorRT 版本号，若无安装 TensorRT, 则返回 None 。
    - cuda_archs - 返回 paddle 安装包编译时的 CUDA 架构列表。

代码示例
::::::::::

COPY-FROM: paddle.version.show
