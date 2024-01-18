.. _cn_api_paddle_jit_save:

save
-----------------

.. py:function:: paddle.jit.save(layer, path, input_spec=None, **configs)

将输入的 ``Layer`` 或 ``function`` 存储为 ``paddle.jit.TranslatedLayer`` 格式的模型，载入后可用于预测推理或者 fine-tune 训练。

该接口会将输入 ``Layer`` 转写后的模型结构 ``Program`` 和所有必要的持久参数变量存储至输入路径 ``path`` 。

``path`` 是存储目标的前缀，存储的模型结构 ``Program`` 文件的后缀为 ``.pdmodel`` ，存储的持久参数变量文件的后缀为 ``.pdiparams``，同时这里也会将一些变量描述信息存储至文件，文件后缀为 ``.pdiparams.info``，这些额外的信息将在 fine-tune 训练中使用。

存储的模型能够被以下 API 完整地载入使用：

    - ``paddle.jit.load``
    - ``paddle.static.load_inference_model``
    - 其他 C++ 预测库 API

.. note::
    当使用 ``paddle.jit.save`` 保存 ``function`` 时，``function`` 不能包含参数变量。如果必须保存参数变量，请用 Layer 封装 function，然后按照处理 Layer 的方式调用相应的 API。

参数
:::::::::
    - **layer** (Layer|function) - 需要存储的 ``Layer`` 对象或者 ``function``。
    - **path** (str) - 存储模型的路径前缀。格式为 ``dirname/file_prefix`` 或者 ``file_prefix`` 。
    - **input_spec** (list or tuple[InputSpec|Tensor|Python built-in variable]，可选) - 描述存储模型 forward 方法的输入，可以通过 InputSpec 或者示例 Tensor 进行描述。此外，我们还支持指定非张量类型的参数，比如 int、float、string，或者这些类型的列表/字典。如果为 ``None``，所有原 ``Layer`` forward 方法的输入变量将都会被配置为存储模型的输入变量。默认为 ``None``。
    - **configs** (dict，可选) - 其他用于兼容的存储配置选项。这些选项将来可能被移除，如果不是必须使用，不推荐使用这些配置选项。默认为 ``None``。目前支持以下配置选项：(1) output_spec (list[Tensor]) - 选择存储模型的输出目标。默认情况下，所有原 ``Layer`` forward 方法的返回值均会作为存储模型的输出。如果传入的 ``output_spec`` 列表不是所有的输出变量，存储的模型将会根据 ``output_spec`` 所包含的结果被裁剪。

返回
:::::::::
无

代码示例
:::::::::

COPY-FROM: paddle.jit.save
