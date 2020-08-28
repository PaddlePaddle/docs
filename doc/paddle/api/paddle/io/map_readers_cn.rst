.. _cn_api_fluid_io_map_readers:

map_readers
-------------------------------

.. py:function::   paddle.fluid.io.map_readers(func, *readers)




该接口将创建一个数据读取器(Reader)，其中 `func` 函数的输出将直接作为新数据读取器的输出， `readers` 的输出将作为函数 `func` 的输入参数。

例如：如果输入的 `readers` 为两个输出分别为：2、3 的 `reader` ，输入的 `func` 为乘法函数 `mul(x, y)` ，则得到的新建 `reader` 的输出为：6。

参数：
    - **func**  - 读取数据并返回数据项的函数， `func` 的输出将直接作为新创建的数据读取器的输出。 

    - **readers** - 输入的一个或多个数据读取器(Reader)，这些数据读取器的输出数据将作为函数 `func` 的输入参数。数据读取器的定义参见 :ref:`cn_api_paddle_data_reader_reader` 。
	
返回： 新创建的数据读取器(Reader)

**代码示例**:

.. code-block:: python


   import paddle.fluid as fluid
   d = {"h": 0, "i": 1}
   def func(x):
       return d[x]

   def reader():
       yield "h"
       yield "i"

   map_reader_result = fluid.io.map_readers(func, reader)




