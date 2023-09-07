.. _cn_api_paddle_load:

load
-----

.. py:function:: paddle.load(path, **configs)

从指定路径载入可以在 paddle 中使用的对象实例。

.. note::
    目前支持载入：Layer 或者 Optimizer 的 ``state_dict``，Tensor 以及包含 Tensor 的嵌套 list、tuple、dict、Program。对于 Tensor 对象，只保存了它的名字和数值，没有保存 stop_gradient 等属性，如果您需要这些没有保存的属性，请调用 set_value 接口将数值设置到带有这些属性的 Tensor 中。


遇到使用问题，请参考：

    ..  toctree::
        :maxdepth: 1

        ../../../../faq/save_cn.md

参数
:::::::::
    - **path** (str|BytesIO) - 载入目标对象实例的路径/内存对象。通常该路径是目标文件的路径，当从用于存储预测模型 API 的存储结果中载入 state_dict 时，该路径可能是一个文件前缀或者目录。
    - **\*\*configs** (dict，可选) - 其他用于兼容的载入配置选项。这些选项将来可能被移除，如果不是必须使用，不推荐使用这些配置选项。默认为 ``None``。目前支持以下配置选项：

        - (1) model_filename (str) - paddle 1.x 版本 ``save_inference_model`` 接口存储格式的预测模型文件名，原默认文件名为 ``__model__`` ；
        - (2) params_filename (str) - paddle 1.x 版本 ``save_inference_model`` 接口存储格式的参数文件名，没有默认文件名，默认将各个参数分散存储为单独的文件；
        - (3) return_numpy(bool) - 如果被指定为 ``True`` ，``load`` 的结果中的 Tensor 会被转化为 ``numpy.ndarray``，默认为 ``False`` 。

返回
:::::::::
Object，一个可以在 paddle 中使用的对象实例。

代码示例 1
:::::::::

COPY-FROM: paddle.load:code-example-1

代码示例 2
:::::::::

COPY-FROM: paddle.load:code-example-2

代码示例 3
:::::::::

COPY-FROM: paddle.load:code-example-3

代码示例 4
:::::::::

COPY-FROM: paddle.load:code-example-4

代码示例 5
:::::::::

COPY-FROM: paddle.load:code-example-5
