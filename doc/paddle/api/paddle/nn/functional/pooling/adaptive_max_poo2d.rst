add
---------
..
  API 名称

API属性：Both
推荐别名：paddle.add
兼容别名：paddle.tensor.add, paddle.tensor.math.add

..
  API 属性与别名
  rst语法
  :api_attr: Both
  :alias_main: paddle.add
  :alias: paddle.tensor.add, paddle.tensor.math.add


``paddle.add(x, y, alpha=1, out=None, name=None)``
..
  API 声明
  rst语法
  .. py:function:: paddle.add(x, y, alpha=1, out=None, name=None)

该API实现``x``与``y``的加法操作。计算公式为：
out = x + y
如果y的维度与x的维度一致，那么会执行element-wise的加法，详情见代码示例 1；
如果y的维度小于x的维度，会执行broadcasting的加法，详情见代码示例 2，代码示例 3；

注解：y的维度不能大于x的维度，否则会报错，详情见示例代码 1中的注释部分。

..
  API 功能描述
  API功能描述部分需要通过文字、公式、图解等多种方式，说明API的实际作用，要以能够让普通用户看懂为目标。参考的维度如下：
  功能作用：描述该API文档的功能作用；如该API用于实现xxxx的功能。
  计算公式：给出该API的计算公式，要给出一些计算的细节，比如说明是不是element-wise的；
  使用场景的示例：给出简单使用以及复杂使用的场景；
  计算细节的一些说明：比如broadcasting的规则；
  与其他API的差异说明：如果该API与其他API功能相似，需要给出该API与另一个API的使用上的区别；

  注意：
  1、文档中的前后说明要一致，比如维度的说明；（错误示范：flatten）
  2、文档相互引用的方式：如何让文档相互引用
  3、功能描述中涉及到的专有数据结构如Tensor与LoDTensor，中英文都直接使用Tensor与LoDTensor，无需翻译。


**x**: (*Variable*) – 输入，任意维度的 ``Tensor`` 或 ``LodTensor``，数据类型支持float32、float64、int32、int64。
**y**: (*Variable*) – 输入，维度小于等于 ``x`` 的 ``Tensor`` 或 ``LodTensor``，数据类型支持float32、float64、int32、int64。
**alpha**: （*int* | *float*，可选）- 对 ``y`` 进行放缩的参数，如果有``alpha``，那么计算公式变为`out = x + alpha * y`，详情见代码示例 4。
**out**: (*Variable*，可选) - 存储运算结果的``Tensor``，默认值为None，表示计算结果的存储不由out控制。
**name**: （str，可选）- 输出的名字。一般无需设置，默认值为None。该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:api_guide_Name 。

..
  API参数
  API参数部分应该描述该API的参数名称，每个参数变量的类型、每个参数支持的数据类型、参数含义以及使用场景，具体要求如下。
  1、完备性要求：参数文档100%覆盖，如果是框架中定义的类型，参数变量类型细化到Variable type。。
  2、一致性要求：参数应和声明部分的参数保持一致，不能有遗漏的情况，还应和描述时的参数名称保持一致；
  3、易读性要求：要解释清楚每个参数的意义和使用场景，对于有默认值的参数，需要分别描述该参数在默认值下的逻辑与非默认值下的逻辑非默认值下的逻辑，而不仅仅是介绍这个参数是什么以及默认值是什么；

  问题：
  参数类型与支持的数据类型分别有多少个？顺序应该是怎样的？


多维``Tensor``或``lodTensor``，维度与数据类型与``x``一致，返回加法操作后的结果。

..
  API 返回
  API的返回部分需要先描述API 的返回值的类型，然后描述API的返回值及其含义。

..
  API抛出异常
  paddle.add 无

代码示例 1

import paddle
import paddle.fluid as fluid
import numpy as np
paddle.enable_imperative()
np_x = np.array([1, 2, 3]).astype('float64')
np_y = np.array([1, 2, 3]).astype('float64')
# np_y = np.array([1, 2, 3, 4]).astype('float64') # 错误，y的维度大于x的维度
x = fluid.dygraph.to_variable(np_x)
y = fluid.dygraph.to_variable(np_y)
z = paddle.add(x, y)
np_z = z.numpy()
print(np_z)  # [2, 4, 6]

代码示例 2
import paddle
import paddle.fluid as fluid
import numpy as np
paddle.enable_imperative()
np_x = np.array([1, 2, 3]).astype('float64')
np_y = np.array([1]).astype('float64')
x = fluid.dygraph.to_variable(np_x)
y = fluid.dygraph.to_variable(np_y)
z = paddle.add(x, y)
np_z = z.numpy()
print(np_z)  # [2, 3, 4]

代码示例 3
import paddle
import paddle.fluid as fluid
import numpy as np
paddle.enable_imperative()
np_x = np.arange(0, 4).reshape((2, 2)).astype('float64')
np_y = np.array([1, 2]).astype('float64')
x = fluid.dygraph.to_variable(np_x)
y = fluid.dygraph.to_variable(np_y)
z = paddle.add(x, y)
np_z = z.numpy()
print(np_z)  # [[1, 3], [3, 5]]

代码示例 4
import paddle
import paddle.fluid as fluid
import numpy as np
paddle.enable_imperative()
np_x = np.array([1, 2, 3]).astype('float64')
np_y = np.array([1]).astype('float64')
x = fluid.dygraph.to_variable(np_x)
y = fluid.dygraph.to_variable(np_y)
z = paddle.add(x, y, alpha=-1)
# alpha=-1, out = x + alpha * y
np_z = z.numpy()
print(np_z)  # [0, 1, 2]

代码示例 5(动态图)

