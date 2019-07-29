.. _cn_api_fluid_CUDAPinnedPlace:

CUDAPinnedPlace
-------------------------------

.. py:class:: paddle.fluid.CUDAPinnedPlace

CUDAPinnedPlace是一个设备描述符，它所指代的存储空间可以被GPU和CPU访问。

**代码示例**

.. code-block:: python

      import paddle.fluid as fluid
      place = fluid.CUDAPinnedPlace()

