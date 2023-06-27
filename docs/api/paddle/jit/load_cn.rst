.. _cn_api_paddle_jit_load:

load
-----------------

.. py:function:: paddle.jit.load(path, **configs)


将接口 ``paddle.jit.save`` 或者 ``paddle.static.save_inference_model`` 存储的模型载入为 ``paddle.jit.TranslatedLayer``，用于预测推理或者 fine-tune 训练。

.. note::
    如果载入的模型是通过 ``paddle.static.save_inference_model`` 存储的，在使用它进行 fine-tune 训练时会存在一些局限：
    1. 命令式编程模式不支持 ``LoDTensor``，所有原先输入变量或者参数依赖于 LoD 信息的模型暂时无法使用；
    2. 所有存储模型的 feed 变量都需要被传入 ``Translatedlayer`` 的 forward 方法；
    3. 原模型变量的 ``stop_gradient`` 信息已丢失且无法准确恢复；
    4. 原模型参数的 ``trainable`` 信息已丢失且无法准确恢复。

参数
:::::::::
    - **path** (str) - 载入模型的路径前缀。格式为 ``dirname/file_prefix`` 或者 ``file_prefix`` 。
    - **config** (dict，可选) - 其他用于兼容的载入配置选项。这些选项将来可能被移除，如果不是必须使用，不推荐使用这些配置选项。默认为 ``None``。目前支持以下配置选项：
        (1) model_filename (str) - paddle 1.x 版本 ``save_inference_model`` 接口存储格式的预测模型文件名，原默认文件名为 ``__model__`` ；
        (2) params_filename (str) - paddle 1.x 版本 ``save_inference_model`` 接口存储格式的参数文件名，没有默认文件名，默认将各个参数分散存储为单独的文件。

返回
:::::::::
TranslatedLayer，一个能够执行存储模型的 ``Layer`` 对象。

代码示例
:::::::::

1. 载入由接口 ``paddle.jit.save`` 存储的模型进行预测推理及 fine-tune 训练。

COPY-FROM: paddle.jit.api.load:code-example1



2. 兼容载入由接口 ``paddle.fluid.io.save_inference_model`` 存储的模型进行预测推理及 fine-tune 训练。

COPY-FROM: paddle.jit.api.load:code-example2
