# 框架概念


## 一、动转静预测部署

动态图由于其与 Python 语法契合的易用性，逐步成为各主流框架的默认模式。但这也带来了在非 Python 环境下的部署问题，需要将动态图的 Python 语句转为可以跨语言、跨平台部署的静态图来部署。

动转静模块**是架在动态图与静态图的一个桥梁**，旨在打破动态图与静态部署的鸿沟，消除部署时对模型代码的依赖，打通与预测端的交互逻辑。

![image](./images/to_static_export.png)



在处理逻辑上，动转静主要包含两个主要模块：

+ **代码层面**：将所有的 Paddle ``layers`` 接口在静态图模式下执行以转为 ``Op`` ，从而生成完整的静态 ``Program``
+ **Tensor层面**：将所有的 ``Parameters`` 和 ``Buffers`` 转为**可导出的 ``Variable`` 参数**（ ``persistable=True`` ）

> 关于动转静模块的具体原理，可以参考 [基本原理](./principle_cn.html)；搭配 `paddle.jit.save` 接口导出预测模型的用法案例，可以参考 [案例解析](./case_analysis_cn.html) 。

如下两小节，将介绍动态图和静态图的概念和差异性，以帮助理解动转静如何起到**桥梁作用**的。

## 二、动态图预测部署

2.0 版本后，Paddle 默认开启了动态图模式。动态图模式下编程组网更加灵活，也更 Pythonic 。在动态图下，模型代码是 **逐行被解释执行** 的。如：

```python
import paddle

zeros = paddle.zeros(shape=[1,2], dtype='float32')
print(zeros)

#Tensor(shape=[1, 2], dtype=float32, place=CPUPlace, stop_gradient=True,
#       [[0., 0.]])
```


**从框架层面上，上述的调用链是：**

> 前端 zeros 接口 &rarr; core.ops.fill_constant (Pybind11)  &rarr; 后端 Kernel  &rarr; 前端 Tensor 输出

如下是一个简单的 Model 示例：

```python

import paddle

class SimpleNet(paddle.nn.Layer):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.linear = paddle.nn.Linear(10, 3)

    def forward(self, x, y):
        out = self.linear(x)
        out = out + y
        return out

net = SimpleNet()
```

动态图下，当实例化一个 ``SimpleNet()`` 对象时，隐式地执行了如下几个步骤：

+ 创建一个 ``Linear`` 对象，记录到 ``self._sub_layer`` 中（dict 类型）

    + 创建一个 ``ParamBase`` 类型的 ``weight`` ，记录到 ``self._parameters`` 中（dict类型）
    + 创建一个 ``ParamBase`` 类型的 ``bias`` ，记录到 ``self._parameters`` 中

一个复杂模型可能包含很多子类，框架层就是通过 ``self._sub_layer`` 和 ``self._parameters`` 两个核心数据结构关联起来的，这也是后续动转静原理上操作的两个核心属性。

```python
sgd = paddle.optimizer.SGD(learning_rate=0.1, parameters=net.parameters())
                                                              ^
                                                              |
                                                         所有待更新参数
```

## 三、静态图预测部署

**静态图编程，总体上包含两个部分：**

+ **编译期**：组合各个 ``Layer`` 接口，搭建网络结构，执行每个 Op 的 ``InferShape`` 逻辑，最终生成 ``Program``
+ **执行期**：构建执行器，输入数据，依次执行每个 ``OpKernel`` ，进行训练和评估

在静态图编译期，变量 ``Variable`` 只是**一个符号化表示**，并不像动态图 ``Tensor`` 那样持有实际数据。

```python
import paddle
# 开启静态图模式
paddle.enable_static()

zeros = paddle.zeros(shape=[1,2], dtype='float32')
print(zeros)
# var fill_constant_1.tmp_0 : LOD_TENSOR.shape(1, 2).dtype(float32).stop_gradient(True)
```

**从框架层面上，静态图的调用链：**

> layer 组网（前端） &rarr; InferShape 检查（编译期） &rarr;  Executor（执行期） &rarr; 逐个执行 OP


如下是 ``SimpleNet`` 的静态图模式下的组网代码：

```python
import paddle
# 开启静态图模式
paddle.enable_static()

# placeholder 信息
x = paddle.static.data(shape=[None, 10], dtype='float32', name='x')
y = paddle.static.data(shape=[None, 3], dtype='float32', name='y')

out = paddle.static.nn.fc(x, 3)
out = paddle.add(out, y)
# 打印查看 Program 信息
print(paddle.static.default_main_program())

# { // block 0
#    var x : LOD_TENSOR.shape(-1, 10).dtype(float32).stop_gradient(True)
#    var y : LOD_TENSOR.shape(-1, 3).dtype(float32).stop_gradient(True)
#    persist trainable param fc_0.w_0 : LOD_TENSOR.shape(10, 3).dtype(float32).stop_gradient(False)
#    var fc_0.tmp_0 : LOD_TENSOR.shape(-1, 3).dtype(float32).stop_gradient(False)
#    persist trainable param fc_0.b_0 : LOD_TENSOR.shape(3,).dtype(float32).stop_gradient(False)
#    var fc_0.tmp_1 : LOD_TENSOR.shape(-1, 3).dtype(float32).stop_gradient(False)
#    var elementwise_add_0 : LOD_TENSOR.shape(-1, 3).dtype(float32).stop_gradient(False)

#    {Out=['fc_0.tmp_0']} = mul(inputs={X=['x'], Y=['fc_0.w_0']}, force_fp32_output = False, op_device = , op_namescope = /, op_role = 0, op_role_var = [], scale_out = 1.0, scale_x = 1.0, scale_y = [1.0], use_mkldnn = False, x_num_col_dims = 1, y_num_col_dims = 1)
#    {Out=['fc_0.tmp_1']} = elementwise_add(inputs={X=['fc_0.tmp_0'], Y=['fc_0.b_0']}, Scale_out = 1.0, Scale_x = 1.0, Scale_y = 1.0, axis = 1, mkldnn_data_type = float32, op_device = , op_namescope = /, op_role = 0, op_role_var = [], use_mkldnn = False, use_quantizer = False, x_data_format = , y_data_format = )
#    {Out=['elementwise_add_0']} = elementwise_add(inputs={X=['fc_0.tmp_1'], Y=['y']}, Scale_out = 1.0, Scale_x = 1.0, Scale_y = 1.0, axis = -1, mkldnn_data_type = float32, op_device = , op_namescope = /, op_role = 0, op_role_var = [], use_mkldnn = False, use_quantizer = False, x_data_format = , y_data_format = )
}
```


静态图中的一些概念：

+ **Program**：与 ``Model`` 对应，描述网络的整体结构，内含一个或多个 ``Block``
+ **Block**
    + **global_block**：全局 ``Block`` ，包含所有 ``Parameters`` 、全部 ``Ops`` 和 ``Variables``
    + **sub_block**：控制流，包含控制流分支内的所有 ``Ops`` 和必要的 ``Variables``
+ **OpDesc**：对应每个前端 API 的计算逻辑描述
+ **Variable**：对应所有的数据变量，如 ``Parameter`` ，临时中间变量等，全局唯一 ``name`` 。

## 三、模型和参数

当训练完一个模型后，下一阶段就是保存导出，实现**模型**和**参数**的分发，进行多端部署。

动态图下，**模型**指的是 Python 前端代码；**参数**指的是 ``model.state_dict()`` 中存放的权重数据。

```python
net = SimpleNet()

# .... 训练过程(略)

layer_state_dict = net.state_dict()
paddle.save(layer_state_dict, "net.pdparams") # 导出模型
```


即意味着，动态图预测部署时，除了已经序列化的参数文件，还须提供**最初的模型组网代码**。

![image](./images/dygraph_export.png)



静态图下，**模型**指的是 ``Program`` ；参数指的是所有的 ``Persistable=True`` 的 ``Variable`` 。二者都可以序列化导出为磁盘文件，**与前端代码完全解耦**。

```python
main_program = paddle.static.default_main_program()

# ...... 训练过程（略）

prog_path='main_program.pdmodel'
paddle.save(main_program, prog_path) # 导出为 .pdmodel

para_path='main_program.pdparams'
paddle.save(main_program.state_dict(), para_path) # 导出为 .pdparams
```

![image](./images/static_export.png)


即意味着， ``Program`` 中包含了模型所有的计算描述（ ``OpDesc`` ），不存在计算逻辑有遗漏的地方。


> 注：更多细节，请参考 [【官方文档】模型的存储与载入](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/02_paddle2.0_develop/08_model_save_load_cn.html)。
