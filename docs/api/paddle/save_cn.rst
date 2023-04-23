.. _cn_api_paddle_framework_io_save:

save
-----

.. py:function:: paddle.save(obj, path, protocol=4)

将对象实例 obj 保存到指定的路径中。

.. note::
    目前支持保存：Layer 或者 Optimizer 的 ``state_dict``，Tensor 以及包含 Tensor 的嵌套 list、tuple、dict，Program。对于 Tensor 对象，只保存了它的名字和数值，没有保存 stop_gradient 等属性，如果您需要这些没有保存的属性，请调用 set_value 接口将数值设置到带有这些属性的 Tensor 中。

.. note::
    不同于 ``paddle.jit.save``，由于 ``paddle.save`` 的存储结果是单个文件，所以不需要通过添加后缀的方式区分多个存储文件，``paddle.save`` 的输入参数 ``path`` 将直接作为存储结果的文件名而非前缀。为了统一存储文件名的格式，我们推荐使用 paddle 标椎文件后缀：
    1. 对于 ``Layer.state_dict``，推荐使用后缀 ``.pdparams`` ；
    2. 对于 ``Optimizer.state_dict``，推荐使用后缀 ``.pdopt`` 。
    具体示例请参考 API 的代码示例。


遇到使用问题，请参考：

    ..  toctree::
        :maxdepth: 1

        ../../../../faq/save_cn.md

参数
:::::::::
 - **obj**  (Object) – 要保存的对象实例。
 - **path**  (str|BytesIO) – 保存对象实例的路径/内存对象。如果存储到当前路径，输入的 path 字符串将会作为保存的文件名。
 - **protocol**  (int，可选) – pickle 模块的协议版本，默认值为 4，取值范围是[2,4]。
 - **configs**  (dict，可选) – 其他配置选项，目前支持以下选项：（1）use_binary_format（bool）- 如果被保存的对象是静态图的 Tensor，你可以指定这个参数。如果被指定为 ``True``，这个 Tensor 会被保存为由 paddle 定义的二进制格式的文件；否则这个 Tensor 被保存为 pickle 格式。默认为 ``False`` 。

返回
:::::::::
无

代码示例 1
:::::::::

COPY-FROM: paddle.save:code-example-1

代码示例 2
:::::::::

COPY-FROM: paddle.save:code-example-2

代码示例 3
:::::::::

COPY-FROM: paddle.save:code-example-3

代码示例 4
:::::::::

COPY-FROM: paddle.save:code-example-4

代码示例 5
:::::::::

COPY-FROM: paddle.save:code-example-5
