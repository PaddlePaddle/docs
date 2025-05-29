.. _cn_api_paddle_utils_try_import:

try_import
-------------------------------

.. py:function:: paddle.utils.try_import(module_name, err_msg=None)

尝试导入一个模块，如果导入失败，提供一条错误信息。

参数
:::::::::
  - **module_name** (str) - 要尝试导入的模块名。
  - **err_msg**  (str, 可选) - 如果导入失败时的自定义错误信息。如果不提供，则会使用默认的错误信息。

返回
:::::::::
如果成功导入，返回对应模块的对象；如果失败，则抛出 ImportError。
