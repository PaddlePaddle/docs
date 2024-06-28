高阶自动微分功能支持科学计算
================

本篇文章主要为你介绍飞桨的高阶微分机制，帮助你更好的使用飞桨。

一、背景与动机
--------

深度学习模型的训练过程涉及使用随机梯度下降（SGD）等优化算法来更新模型参数。在这一过程中，深度学习框架的自动微分功能发挥着核心作用，它利用链式法则自动计算出损失函数相对于模型参数的梯度。尽管大多数深度学习任务只需计算一阶导数，但在某些 AI for Science 场景中，却需要计算高阶导数，这无疑增加了自动微分的复杂性。以 2D 矩形平板分布受载问题为例，该问题的内在机理需要使用 4 阶微分方程来描述。为了求解这类问题，深度学习框架必须支持高阶自动微分功能。

<figure align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/docs/develop/docs/guides/paddle_v3_features/images/overview/paddle_v3_2d_plate.png" style="zoom:100%"/>
</figure>

<figure align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/docs/develop/docs/guides/paddle_v3_features/images/overview/paddle_v3_2d_plate_pde.png" style="zoom:100%"/>
</figure>

二、设计思想
------------------------------

高阶自动微分的实现面临诸多挑战。具体而言，框架需要为每个算子编写高阶微分规则。随着阶数的增加，微分规则的复杂性也随之上升。当阶数达到三阶或更高时，编写这些规则变得极其困难，同时正确性难以保证。为了解决这一问题，飞桨提出了基于基础算子组合的高阶自动微分技术。该技术的关键在于将复杂算子（如 log_softmax）拆解为多个基础算子的组合。然后，我们对这些基础算子进行一阶自动微分变换。重要的是，基础算子经过一阶自动微分变换后，其得到的计算图仍然是由基础算子所构成。通过反复应用一阶自动微分规则，我们可以轻松地获得高阶自动微分结果。

**log_softmax 拆解与微分示例**

根据 log_softmax 表达式拆解为 exp、max、log 等细粒度基础算子组成，基础算子是指由简单运算逻辑组成的有限集合，数量较少。基于飞桨的自动微分体系，使用基础算子的微分规则自动推导 log_softmax 一阶微分，注意基础算子微分规则仍由基础算子实现，因此 log_softmax 的一阶微分仍由基础算子组成。重复上述微分过程实现 log_softmax 高阶微分求解。

<figure align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/docs/develop/docs/guides/paddle_v3_features/images/higher_order_ad/softmax_example.png" style="zoom:80%"/>
</figure>

三、框架架构
------------------------------

为了支持高阶自动微分，飞桨框架精心设计与实现了组合算子机制。这一机制不仅兼容动态图模式和静态图模式，而且在动态图模式下支持 N+1 阶微分的拆分，同时在静态图模式下能够进行编译器融合优化。创新性地设计并实现了动静一体的算子组合规则，这意味着同一套组合规则在动态图和静态图两种模式下均可复用，从而避免了重复开发。在构建基础算子体系时，我们以 Tensor 作为核心操作对象，确保了算子的原子性、实用性和完备性。此外，我们还支持自定义反向操作和自动重计算功能，这些特性不仅提升了模型的精度，还有效地减少了显存占用，为用户提供了更高效、更灵活的深度学习体验。

<figure align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/docs/develop/docs/guides/paddle_v3_features/images/higher_order_ad/architecture.png" style="zoom:80%"/>
</figure>

 **基础算子集合设计**

基础算子集合的设计需要兼顾通用性、计算效率、易用性和兼容性，此外，还需要具备可扩展性，以便可以方便地添加新的数据处理操作和模型，并可以组合支撑更加复杂的计算工作。飞桨制定了基础算子集合设计原则，1）原子性，即基础算子的操作不能拆分为更基础的操作，如不能把大于等于拆分为不小于；2）实用性，基础算子有实际应用场景；3）面向张量，基础算子的操作粒度为张量，如果一个算子需要在张量的元素粒度上进行复杂操作，则这个算子本身应为基础算子；4）完备性，可以支持复杂算子拆分需求。基于上述原则设计和实现基础算子集合，最终预期基础算子规模约控制到 200 左右，当前还在持续演进中。

**动静一体组合规则**

组合规则是指使用基础算子接口组合实现的复杂算子集合，为了能够在动态图和静态图体系下复用同一套组合规则，减少编码工作量，在基础算子层，设计一套抽象接口，屏蔽动态图基础算子和静态图基础算子实现细节，组合规则的实现调用抽象接口实现，并设计一套分发机制，根据动态图和静态图数据类型的不同进行分发到具体基础算子执行，从而实现动态图和静态图不同模式下组合规则的复用。

**从机制上保障性能**

随着算子细粒度拆分，算子数量会急剧膨胀，算子调度开销也会加大。动态图模式算子动态执行，无法提前优化，为了减少算子拆分造成的动态图性能损耗，飞桨采取了拆解 N+1 阶算子方法，即如果现有算子已经实现了 N 阶反向大算子，那么为了保证现有模型性能不降，实现 N+1 拆解逻辑，从而调度上优先运行 1-N 阶大算子逻辑，N+1 拆解成基础算子，保证性能同时支持高阶微分。静态图模式下，由于可以提前整图优化，基于飞桨编译器技术进行图层调度优化和算子融合优化，并且由于算子粒度更细，存在优化空间更大，部分模型上基于组合算子体系和编译器优化的模型性能已经超越了原有大算子体系下模型性能。

**从机制上保障显存和精度**

模型执行过程通常是先执行前向计算，并保存反向计算依赖的中间变量，反向计算复用这些中间变量进行计算。算子细粒度拆分，使需要保存的中间变量急剧增大，模型运行需要的显存大幅增加。飞桨使用自定义反向技术解决该问题，对于一个复杂大算子，支持自定义其反向微分规则，该微分规则实现只依赖其前向大算子的输入输出，并在框架调度上优先保障走该算子的自定义反向微分，而非自动推导的微分规则，从而减少中间变量，降低显存。



四、开始使用
------------------------------

飞桨提供了完善高阶自动微分求解 API，包括通用反向微分求解 paddle.grad，多元函数雅可比矩阵计算 `paddle.autograd.jacobian` ，多元函数海森矩阵计算 `paddle.autograd.hessian`. 功能与链接具体参考 4.1.

下面通过一个简单示例演示飞桨高阶自动微分用法。

**第一步：导入依赖**

```python
import paddle
```

**第二步：编写组网代码**

以单层的全联接网络为例，MyNet 继承自 paddle.nn.Layer，在__init__方法中初始化网络参数，在 forward 方法中实现前向运行逻辑。注意，当前高阶自动微分支持大部分飞桨常用 API，覆盖主流的科学计算模型，如果您在写新的模型遇到飞桨高阶微分问题，可通过飞桨 ISSUE 反馈。

```python
class MyNet(paddle.nn.Layer):
    def __init__(self):
        super(MyNet, self).__init__()
        self.weight = self.create_parameter(shape=(2,2), dtype=paddle.float32, is_bias=False)
        self.bias = self.create_parameter(shape=(2,2), dtype=paddle.float32, is_bias=True)
        self.add_parameter("weight", self.weight)
        self.add_parameter("bias", self.bias)

    def forward(self, x):
        y = paddle.matmul(x, self.weight) + self.bias
        return paddle.tanh(y)
```

**第三步：创建网络及声明输入数据，执行前向计算过程**

```python
x = paddle.randn(shape=(2,2), dtype=paddle.float32)
net = MyNet()
y = net(x)
```

**第四步：计算 Loss**

为了演示高阶微分用法，此处 Loss 定义中使用了`paddle.grad` API 计算`y`对`x`二阶微分，使用`L2 norm` 归一化。

```python
grad1 = paddle.grad(y, x)
grad2 = paddle.grad(grad1, x)
loss = paddle.norm(grad2, p=2)

opt = paddle.optimizer.Adam(0.01)
opt.update(loss)
```

**第五步：执行反向计算过程，使用用 Adam 优化器更新参数**

```python
opt = paddle.optimizer.Adam(parameters=net.parameters())
loss.backward()
opt.step()
```




### 4.1 自动微分相关 API 列表


API 名称 | API 功能 |
:-----: | :-----: |
[paddle.grad](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/grad_cn.html#grad) | 反向模式自动微分 |
[paddle.auto.jacobian](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/autograd/jacobian_cn.html#jacobian) | 雅可比矩阵计算 |
[paddle.autograd.hessian](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/autograd/hessian_cn.html#hessian) | 海森矩阵计算 |



**使用反向微分 API paddle.grad 计算 tanh 高阶导数**


```python
import paddle

# 组网代码
x = paddle.rand((2,))
y = paddle.tanh(x)
grad1 = paddle.grad(y, x, create_graph=True)     # 一阶微分
grad2 = paddle.grad(grad1, x, create_graph=True) # 二阶微分
grad3 = paddle.grad(grad2, x) # 三阶微分

print(grad1, grad2, grad3)
# [0.41997433] [-0.6397] [0.6216267]
```

**使用 paddle.autograd.jacobian 计算 Jacobian 矩阵**

```python
import paddle

x1 = paddle.randn([3, ])
x2 = paddle.randn([3, ])
x1.stop_gradient = False
x2.stop_gradient = False

y = x1 + x2

J = paddle.autograd.jacobian(y, (x1, x2))
J_y_x1 = J[0][:] # evaluate result of dy/dx1
J_y_x2 = J[1][:] # evaluate result of dy/dx2

print(J_y_x1.shape)
# [3, 3]
print(J_y_x2.shape)
# [3, 3]
```

**使用 paddle.autograd.hessian 计算 Hessian 矩阵**

```python
import paddle

x1 = paddle.randn([3, ])
x2 = paddle.randn([4, ])
x1.stop_gradient = False
x2.stop_gradient = False

y = x1.sum() + x2.sum()

H = paddle.autograd.hessian(y, (x1, x2))
H_y_x1_x1 = H[0][0][:] # evaluate result of ddy/dx1x1
H_y_x1_x2 = H[0][1][:] # evaluate result of ddy/dx1x2
H_y_x2_x1 = H[1][0][:] # evaluate result of ddy/dx2x1
H_y_x2_x2 = H[1][1][:] # evaluate result of ddy/dx2x2

print(H_y_x1_x1.shape)
# [3, 3]
print(H_y_x1_x2.shape)
# [3, 4]
print(H_y_x2_x1.shape)
# [4, 3]
print(H_y_x2_x2.shape)
# [4, 4]
```



五、飞桨支撑科学计算 AI For Science
------------------------------

基于飞桨框架 3.0 为科学计算提供了高阶自动微分、编译优化、分布式训练能力支撑，提供了面向通用数理问题求解的赛桨 PaddleScience 以及专注于生物计算的螺旋桨 PaddleHelix 工具组件。为了更好地支撑 AI for Science 生态，飞桨对国内外主流开源科学计算工具进行了适配，并被国际主流的科学计算深度学习库 DeepXDE 唯一推荐。在与 NVIDIA 合作适配其 AI Physics 工具 Modulus 的过程中，飞桨利用其高阶自动微分与编译优化技术，成功完成了全量模型适配，实现了方程求解类模型性能的大幅优化，相比 Modulus 现有后端求解速度平均提升 71%。

<figure align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/docs/develop/docs/guides/paddle_v3_features/images/higher_order_ad/ai4s.png" style="zoom:40%"/>
</figure>
