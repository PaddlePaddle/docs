# 高阶自动微分功能支持科学计算

本篇文章主要为你介绍飞桨的高阶微分机制，帮助你更好的使用飞桨。

## 一、背景与动机

深度学习模型的训练过程涉及使用随机梯度下降（SGD）等优化算法来更新模型参数。在这一过程中，深度学习框架的自动微分功能发挥着核心作用，它利用链式法则自动计算出损失函数相对于模型参数的梯度。尽管大多数深度学习任务只需计算一阶导数，但在某些 AI for Science 场景中，却需要计算高阶导数，这无疑增加了自动微分的复杂性。以 2D 矩形平板分布受载问题为例，该问题的内在机理需要使用 4 阶微分方程来描述。为了求解这类问题，深度学习框架必须支持高阶自动微分功能。

<figure align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/docs/develop/docs/guides/paddle_v3_features/images/overview/paddle_v3_2d_plate.png" style="zoom:100%"/>
</figure>

<figure align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/docs/develop/docs/guides/paddle_v3_features/images/overview/paddle_v3_2d_plate_pde.png" style="zoom:100%"/>
</figure>

## 二、设计思想

高阶自动微分的实现面临诸多挑战。具体而言，框架需要为每个算子编写高阶微分规则。随着阶数的增加，微分规则的复杂性也随之上升。当阶数达到三阶或更高时，编写这些规则变得极其困难，同时正确性难以保证。为了解决这一问题，飞桨提出了基于基础算子组合的高阶自动微分技术。该技术的关键在于将复杂算子（如 `log_softmax`）拆解为多个基础算子的组合。然后，我们对这些基础算子进行一阶自动微分变换。重要的是，基础算子经过一阶自动微分变换后，其得到的计算图仍然是由基础算子所构成。通过反复应用一阶自动微分规则，我们可以轻松地获得高阶自动微分结果。

**log_softmax 拆解与微分示例**

`log_softmax` 计算过程，可拆解为 `exp`、`max`、`log` 等细粒度的基础算子（基础算子是指由简单、不可再拆分运算逻辑组成的有限集合，数量较少）。而后基于飞桨的自动微分体系，即可使用使用基础算子的微分规则自动推导出 `log_softmax` 的一阶反向微分。又由于基础算子微分规则仍由基础算子实现，因此 `log_softmax` 的二阶以及更高阶的微分仍然由基础算子组成。综上所述，通过基础算子微分规则的组合，即可实现复杂算子的高阶微分。

<figure align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/docs/develop/docs/guides/paddle_v3_features/images/higher_order_ad/softmax_example.png" style="zoom:80%"/>
</figure>

## 三、框架架构

为了支持高阶自动微分，飞桨框架精心设计与实现了组合算子机制。这一机制不仅兼容动态图模式和静态图模式，而且在动态图模式下支持 N+1 阶微分的拆分(即第 N 阶微分使用手写算子，第 N+1 阶微分使用组合算子逻辑，最大限度保证存量算子和模型的精度、性能)，同时在静态图模式下能够进行编译器融合优化。创新性地设计并实现了动静一体的算子组合规则，这意味着同一套组合规则在动态图和静态图两种模式下均可复用，从而避免了重复开发。在构建基础算子体系时，我们以 Tensor 作为核心操作对象，确保了算子的原子性、实用性和完备性。此外，我们还支持自定义反向操作和自动重计算功能，这些特性不仅提升了模型的精度，还有效地减少了显存占用，为用户提供了更高效、更灵活的深度学习体验。

<figure align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/docs/develop/docs/guides/paddle_v3_features/images/higher_order_ad/architecture.png" style="zoom:80%"/>
</figure>

- 基础算子集合设计

基础算子集合的设计需要兼顾通用性、计算效率、易用性和兼容性，此外，还需要具备可扩展性，以便可以方便地添加新的数据处理操作和模型，并可以组合支撑更加复杂的计算工作。飞桨制定了基础算子集合设计原则，1）原子性，即基础算子的操作不能拆分为更基础的操作，如不能把大于等于拆分为不小于；2）实用性，基础算子有实际应用场景；3）面向张量，基础算子的操作粒度为张量，如果一个算子需要在张量的元素粒度上进行复杂操作，则这个算子本身应为基础算子；4）完备性，可以支持复杂算子拆分需求。基于上述原则设计和实现基础算子集合，最终预期基础算子规模约控制到 200 左右，当前还在持续演进中。

- 动静一体组合规则

组合规则是指使用基础算子接口组合实现的复杂算子集合，为了能够在动态图和静态图体系下复用同一套组合规则，减少编码工作量，在基础算子层，设计一套抽象接口，屏蔽动态图基础算子和静态图基础算子实现细节，组合规则的实现调用抽象接口实现，并设计一套分发机制，根据动态图和静态图数据类型的不同进行分发到具体基础算子执行，从而实现动态图和静态图不同模式下组合规则的复用。

- 从机制上保障性能

随着算子细粒度拆分，算子数量会急剧膨胀，算子调度开销也会加大。动态图模式算子动态执行，无法提前优化，为了减少算子拆分造成的动态图性能损耗，飞桨采取了拆解 N+1 阶算子方法。即如果现有算子已经实现了 N 阶反向大算子，为了保证现有模型性能不降低，会实现第 N+1 阶算子的拆解逻辑。在调度上优先运行 1~N 阶大算子，在第 N+1 阶才会拆解成基础算子，保证性能同时支持高阶微分。静态图模式下，由于可以提前整图优化，基于飞桨编译器技术进行图层调度优化和算子融合优化，并且由于算子粒度更细，存在优化空间更大，部分模型上基于组合算子体系和编译器优化的模型性能已经超越了原有大算子体系下模型性能。

- 从机制上保障显存和精度

模型执行过程通常是先执行前向计算，并保存反向计算依赖的中间变量，反向计算复用这些中间变量进行计算。算子细粒度拆分，使需要保存的中间变量急剧增大，模型运行需要的显存大幅增加。飞桨使用自定义反向技术解决该问题，对于一个复杂大算子，支持自定义其反向微分规则，该微分规则实现只依赖其前向大算子的输入输出，并在框架调度上优先保障走该算子的自定义反向微分，而非自动推导的微分规则，从而减少中间变量，降低显存。

## 四、二维平板分布受载问题

### 4.1 问题描述

基于上述飞桨的高阶自动微分能力，接下来尝试解决在第一章提到的“2D 矩形平板分布受载问题”。首先列出该问题的数学模型：

薄板小挠度理论的基本方程为：
$$
\frac{\partial^4 w}{\partial x^4}+2 \frac{\partial^4 w}{\partial x^2 \partial y^2}+\frac{\partial^4 w}{\partial y^4}=\frac{q}{D}
$$

其中 $w(x,y)$ 表示薄板的挠度，即薄板在垂直载荷作用下的变形或偏移量，$x,y$ 表示薄板在平面内的坐标，$D$ 为薄板的弯曲刚度，$q$ 是作用在薄板上的面载荷，表示每单位面积上的外部载荷。

在本问题中，矩形薄板 $x$ 方向长 $2m$，$y$ 方向宽 $1m$，板厚 $10mm$，$x$ 方向左右两边处于简支状态（可以转动但不能位移），$y$ 方向上下两边自由（没有任何约束，可以自由移动和转动）。

左右两边 $(x=-1 \mid x=+1)$ 为简支边界条件，因此挠度 $w$ 和弯矩 $M_x$ 都为 $0$ :

$$
(w)\_{x=-1 \mid x=+1}=0, \quad\left(M\_x\right)\_{x=-1 \mid x=+1}=0
$$

由于 $M_x=-D\left(\frac{\partial^2 w}{\partial x^2}+\mu \frac{\partial^2 w}{\partial y^2}\right)$， 且 $\frac{\partial^2 w}{\partial y^2}=0$， 所以简支边界条件可化简为：

$$
(w)\_{x=-1 \mid x=+1}=0, \quad\left(\frac{\partial^2 w}{\partial x^2}\right)\_{x=-1 \mid x=+1}=0
$$

上下两边 $(y=-0.5 \mid y=+0.5)$ 为自由边界条件， 弯矩、扭矩、横向剪切力都为 $0$ :

$$
\left(M\_y\right)\_{\mathrm{y}=-0.5 \mid \mathrm{y}=+0.5}=0, \quad\left(M\_{x y}\right)\_{\mathrm{y}=-0.5 \mid \mathrm{y}=+0.5}=0, \quad\left(Q\_y\right)\_{\mathrm{y}=-0.5 \mid \mathrm{y}=+0.5}=0
$$

由于 $M_y=-D\left(\frac{\partial^2 w}{\partial y^2}+\mu \frac{\partial^2 w}{\partial x^2}\right), \quad M_{x y}=-D(1-\mu) \frac{\partial^2 w}{\partial x \partial y}, \quad Q_y=-D \frac{\partial}{\partial y}\left(\frac{\partial^2 w}{\partial x^2}+\frac{\partial^2 w}{\partial y^2}\right)$ ，且扭矩可以变换为等效剪力， 扭矩和横向剪力合并为 $\left(Q_y+\frac{\partial M_{x y}}{\partial x}\right)_{\mathrm{y}=-0.5 \mid \mathrm{y}=+0.5}=0$， 所以自由边界条件用挠度表示为

$$
\left(\frac{\partial^2 w}{\partial y^2}+\mu \frac{\partial^2 w}{\partial x^2}\right)\_{y=-0.5 \mid y=+0.5}=0, \quad\left(\frac{\partial^3 w}{\partial y^3}+(2-\mu) \frac{\partial^3 w}{\partial x^2 \partial y}\right)\_{y=-0.5 \mid y=+0.5}=0
$$

### 4.2 使用飞桨原生 API 求解

接下来给出上述问题转换成的飞桨代码。

```python
import paddle
import numpy as np
from paddle import grad
from matplotlib import pyplot as plt

# 设置薄板计算域长、宽参数
Lx = 2.0  # 薄板 x 方向长度(m)
Ly = 1.0  # 薄板 y 方向宽度(m)

# 设置方程参数
E = 210000.0e6  # 弹性模量(Pa)
mu = 0.28  # 薄板泊松比(无量纲)
h = 0.01  # 薄板厚度(m)
D = E * (h**3) / (12 * (1 - mu**2))  # 薄板弯曲刚度(kN*m^2)
q = 1000.0  # 均布载荷(N/m^2)

in_channels = 2  # 输入为 x, y
out_channels = 1  # 输出为 w

model = paddle.nn.Sequential(
    paddle.nn.Linear(in_features=in_channels, out_features=32),
    paddle.nn.Silu(),
    paddle.nn.Linear(in_features=32, out_features=64),
    paddle.nn.Silu(),
    paddle.nn.Linear(in_features=64, out_features=32),
    paddle.nn.Silu(),
    paddle.nn.Linear(in_features=32, out_features=out_channels),
)  # 建立一个简单的多层全连接层模型，包含 4 个线性层和 3 个激活函数。

opt = paddle.optimizer.Adam(5e-4, parameters=model.parameters())


def grad_with_order(y, x, k):
    # Compute the gradient of y with respect to x at order k.
    g = y
    for _ in range(k):
        g = grad(g, x, create_graph=True)[0]

    return g


def mse(pred, label):
    # Compute the mean squared error between the predicted and true values.
    return ((pred - label) ** 2).mean()


for i in range(1000):
    # 1. define pde loss
    np_rand_xy = np.random.uniform(
        [-Lx / 2, -Ly / 2], [Lx / 2, Ly / 2], size=(1000, 2)
    ).astype("float32")
    x = paddle.to_tensor(np_rand_xy[:, 0:1], stop_gradient=False)  # [1000, 1]
    y = paddle.to_tensor(np_rand_xy[:, 1:2], stop_gradient=False)  # [1000, 1]
    tensor_input = paddle.concat([x, y], axis=1)  # [1000, 2]
    output = model(tensor_input)
    w = output
    w_x_x = grad_with_order(w, x, 2)
    w_y_y = grad_with_order(w, y, 2)
    w_x_x_x_x = grad_with_order(w_x_x, x, 2)
    w_y_y_y_y = grad_with_order(w_y_y, y, 2)
    w_x_x_y_y = grad_with_order(w_x_x, y, 2)

    loss_pde = mse(w_x_x_x_x + 2 * w_x_x_y_y + w_y_y_y_y, q / D)

    # 2. define bc_left_right_loss
    np_rand_x = np.random.choice([-Lx / 2, Lx / 2], size=(50, 1)).astype("float32")
    np_rand_y = np.random.uniform(-Ly / 2, Ly / 2, size=(50, 1)).astype("float32")
    x = paddle.to_tensor(np_rand_x, stop_gradient=False)  # [50, 1]
    y = paddle.to_tensor(np_rand_y, stop_gradient=False)  # [50, 1]
    tensor_input = paddle.concat([x, y], axis=1)  # [50, 2]
    output = model(tensor_input)
    w = output
    w_x_x = grad_with_order(w, x, 2)

    loss_bc_lr = mse(w, 0) + mse(w_x_x, 0)

    # 3. define bc_loss
    np_rand_x = np.random.uniform(-Lx / 2, Lx / 2, size=(50, 1)).astype("float32")
    np_rand_y = np.random.choice([-Ly / 2, Ly / 2], size=(50, 1)).astype("float32")
    x = paddle.to_tensor(np_rand_x, stop_gradient=False)  # [50, 1]
    y = paddle.to_tensor(np_rand_y, stop_gradient=False)  # [50, 1]
    tensor_input = paddle.concat([x, y], axis=1)  # [50, 2]
    output = model(tensor_input)
    w = output
    w_x_x = grad_with_order(w, x, 2)
    w_y_y = grad_with_order(w, y, 2)
    w_y_y_y = grad_with_order(w_y_y, y, 1)
    w_x_x_y = grad_with_order(w_x_x, y, 1)
    loss_bc_ud = mse(w_y_y + mu * w_x_x, 0) + mse(w_y_y_y + (2 - mu) * w_x_x_y, 0)

    # loss backward and update parameters
    loss = loss_pde + loss_bc_lr + loss_bc_ud
    opt.clear_grad()
    loss.backward()
    opt.step()
    if i % 10 == 0:
        print(f"Loss at iter {i}: {loss.item():.3e}")


# plot result
num_cord0 = 101
num_cord1 = 101
num_cords = num_cord0 * num_cord1
print(f"num_cords = {num_cords}")
x, y = np.meshgrid(
    np.linspace(
        start=-Lx / 2, stop=Lx / 2, num=num_cord0, endpoint=True, dtype="float32"
    ),
    np.linspace(
        start=-Ly / 2, stop=Ly / 2, num=num_cord1, endpoint=True, dtype="float32"
    ),
)
x = x.ravel()
y = y.ravel()
# predict solution of w(x, y) on the 2D grid
w_pred = model(paddle.stack((paddle.to_tensor(x), paddle.to_tensor(y)), axis=1))
w_pred = w_pred.numpy()
fig = plt.figure(100, figsize=(5, 4))
y_min = w_pred.min(axis=(0,))[0]
y_max = w_pred.max(axis=(0,))[0]
ax1 = plt.subplot(1, 1, 1)
plt.tricontourf(x, y, w_pred[:, 0], levels=30, cmap="rainbow")
print(x.shape, y.shape, w_pred.shape)
cb1 = plt.colorbar()
plt.axis("equal")
plt.xlabel("$x (m)$")
plt.ylabel("$y (m)$")
plt.title(f"w-field: [{y_min:.6f}, {y_max:.6f}]", fontsize=9.5)
# plt.show()
plt.savefig("./result.jpg")
print("saved matplotlib to: ./result.jpg")
```

### 4.3 使用 PaddleScience API 求解

基于飞桨框架，我们开发了科学计算套件 [**PaddleScience**](https://paddlescience-docs.readthedocs.io/zh-cn/latest/)，并提供了更上层的 API: [`ppsci.lambdify`](https://paddlescience-docs.readthedocs.io/zh-cn/latest/zh/api/utils/symbolic/?h=#ppsci.utils.symbolic.lambdify)。`ppsci.lambdify` 可以自动将 sympy 表达式转换为基于 Paddle 原生 API 的计算函数，从而避免用户多次显式调用 `paddle.grad`，同时该 API 具备子表达式缓存机制，使得用户无需再关注中间变量的复用。

```python
import paddle
import numpy as np
from matplotlib import pyplot as plt
import ppsci
import sympy as sp

# 设置薄板计算域长、宽参数
Lx = 2.0  # 薄板 x 方向长度(m)
Ly = 1.0  # 薄板 y 方向宽度(m)

# 设置方程参数
E = 210000.0e6  # 弹性模量(Pa)
mu = 0.28  # 薄板泊松比(无量纲)
h = 0.01  # 薄板厚度(m)
D = E * (h**3) / (12 * (1 - mu**2))  # 薄板弯曲刚度(kN*m^2)
q = 1000.0  # 均布载荷(N/m^2)

in_channels = 2  # 输入为 x, y
out_channels = 1  # 输出为 w

model = ppsci.arch.MLP(
    ["x", "y"], ["w"], num_layers=None, hidden_size=[32, 64, 32], activation="silu"
)

opt = paddle.optimizer.Adam(5e-4, parameters=model.parameters())


def mse(pred, label):
    # Compute the mean squared error between the predicted and true values.
    return ((pred - label) ** 2).mean()


# 使用 sympy 库计算符号公式
x, y = sp.symbols("x y")  # 定义符号变量 x, y
w = sp.Function("w")(x, y)  # 定义函数 w(x,y)
left = (
    w.diff(x, 4) + 2 * w.diff(x, 2).diff(y, 2) + w.diff(y, 4)
)  # 定义薄板弯曲的双调和方程的左侧部分
bc_lr = w.diff(x, 2)
bc_ud1 = w.diff(y, 2) + mu * w.diff(x, 2)
bc_ud2 = w.diff(y, 3) + (2 - mu) * w.diff(x, 2).diff(y)

# lambdify the sympy expression to a callable function
pde_func = ppsci.lambdify(left, model)
bc_lr_func = ppsci.lambdify(bc_lr, model)
bc_ud1_func, bc_ud2_func = ppsci.lambdify([bc_ud1, bc_ud2], model)

for i in range(1000):
    # 1. define pde loss
    np_rand_xy = np.random.uniform(
        [-Lx / 2, -Ly / 2], [Lx / 2, Ly / 2], size=(1000, 2)
    ).astype("float32")
    x = paddle.to_tensor(np_rand_xy[:, 0:1], stop_gradient=False)  # [1000, 1]
    y = paddle.to_tensor(np_rand_xy[:, 1:2], stop_gradient=False)  # [1000, 1]
    data_dict = {"x": x, "y": y}
    pde_out = pde_func(data_dict)
    loss_pde = mse(pde_out, q / D)

    # 2. define bc_left_right_loss
    np_rand_x = np.random.choice([-Lx / 2, Lx / 2], size=(50, 1)).astype("float32")
    np_rand_y = np.random.uniform(-Ly / 2, Ly / 2, size=(50, 1)).astype("float32")
    x = paddle.to_tensor(np_rand_x, stop_gradient=False)  # [50, 1]
    y = paddle.to_tensor(np_rand_y, stop_gradient=False)  # [50, 1]
    data_dict = {"x": x, "y": y}
    w_x_x = bc_lr_func(data_dict)
    loss_bc_lr = mse(data_dict["w"], 0) + mse(w_x_x, 0)

    # 3. define bc_loss
    np_rand_x = np.random.uniform(-Lx / 2, Lx / 2, size=(50, 1)).astype("float32")
    np_rand_y = np.random.choice([-Ly / 2, Ly / 2], size=(50, 1)).astype("float32")
    x = paddle.to_tensor(np_rand_x, stop_gradient=False)  # [50, 1]
    y = paddle.to_tensor(np_rand_y, stop_gradient=False)  # [50, 1]
    data_dict = {"x": x, "y": y}
    bc_ud1_out = bc_ud1_func(data_dict)
    bc_ud2_out = bc_ud2_func(data_dict)
    loss_bc_ud = mse(bc_ud1_out, 0) + mse(bc_ud2_out, 0)

    # loss backward and update parameters
    loss = loss_pde + loss_bc_lr + loss_bc_ud
    opt.clear_grad()
    loss.backward()
    opt.step()
    if i % 10 == 0:
        print(f"Loss at iter {i}: {loss.item():.3e}")

# plot result
num_cord0 = 101
num_cord1 = 101
num_cords = num_cord0 * num_cord1
print(f"num_cords = {num_cords}")
x, y = np.meshgrid(
    np.linspace(
        start=-Lx / 2, stop=Lx / 2, num=num_cord0, endpoint=True, dtype="float32"
    ),
    np.linspace(
        start=-Ly / 2, stop=Ly / 2, num=num_cord1, endpoint=True, dtype="float32"
    ),
)
x = x.ravel()
y = y.ravel()
# predict solution of w(x, y) on the 2D grid
w_pred = model.forward_tensor(
    paddle.stack((paddle.to_tensor(x), paddle.to_tensor(y)), axis=1)
)
w_pred = w_pred.numpy()
fig = plt.figure(100, figsize=(5, 4))
y_min = w_pred.min(axis=(0,))[0]
y_max = w_pred.max(axis=(0,))[0]
ax1 = plt.subplot(1, 1, 1)
plt.tricontourf(x, y, w_pred[:, 0], levels=30, cmap="rainbow")
print(x.shape, y.shape, w_pred.shape)
cb1 = plt.colorbar()
plt.axis("equal")
plt.xlabel("$x (m)$")
plt.ylabel("$y (m)$")
plt.title(f"w-field: [{y_min:.6f}, {y_max:.6f}]", fontsize=9.5)
# plt.show()
plt.savefig("./result.jpg")
print("saved matplotlib to: ./result.jpg")
```

## 五、飞桨支撑科学计算 AI4S

基于飞桨框架 3.0 为科学计算提供了高阶自动微分、编译优化、分布式训练能力支撑，提供了面向通用数理问题求解的赛桨 [**PaddleScience**](https://paddlescience-docs.readthedocs.io/zh-cn/latest/) 以及专注于生物计算的螺旋桨 [**PaddleHelix**](https://paddlehelix.baidu.com/) 工具组件。为了更好地支撑 AI for Science 生态，飞桨对国内外主流开源科学计算工具进行了适配，并被国际主流的科学计算深度学习库 DeepXDE 唯一推荐。

### 5.1 飞桨 + Modulus-sym

飞桨利用高阶自动微分与编译优化技术，在与 NVIDIA 合作适配其 AI Physics 工具 Modulus-sym 的过程中，成功完成了全量模型适配([**Modulus-sym(paddle-backend)**](https://github.com/PaddlePaddle/modulus-sym/tree/paddle?tab=readme-ov-file#modulus-symbolic-betapaddle-backend))，实现了方程求解类模型性能的大幅优化，相比 Modulus-sym 现有后端**求解速度平均提升 115%**；

![ai4s.png](https://raw.githubusercontent.com/PaddlePaddle/docs/develop/docs/guides/paddle_v3_features/images/higher_order_ad/ai4s.png)

> 上述测试环境为：cuda 11.8, A100-SXM4-40GB, torch 2.6(2236df1), paddle 3.0(388165), ips = total_batch_size / batch_cost(ms)

### 5.2 飞桨 + DeePMD-kit

在 AI 分子动力学套件 [**DeePMD-kit**](https://docs.deepmodeling.com/projects/deepmd/en/latest/train/training.html) 中，我们对 dpa2, se_atten, se_e2_a 进行了动态图和编译器适配，相比 DeePMD-kit torch 后端，**求解速度分别提升了 102.6%, 40.5%, 102.6%**，相关结果已公开至论文：[DeePMD-kit v3: A Multiple-Backend Framework for Machine Learning Potentials](https://arxiv.org/abs/2502.19161)。

基于飞桨后端运行 DeePMD-kit 可参考：[5.1. Train a model](https://docs.deepmodeling.com/projects/deepmd/en/latest/train/training.html)

| 模型名称/平均耗时(s/batch) | Torch(dygraph) | Paddle(dygraph) | Paddle(CINN) | IPS 提升率 |
|:---------------------------|:---------------|:----------------|:-------------|:-----------|
| dpa2                       | 0.1064         | 0.120           | **0.053**    | 102.6%     |
| se_atten                   | 0.0336         | 0.049           | **0.024**    | 40.5%      |
| se_e2_a                    | 0.0227         | 0.025           | **0.011**    | 102.6%     |

> 上述测试环境为：cuda 11.8, A100-SXM4-40GB, torch 2.6(2236df1), paddle 3.0(86994e3), ips 提升率 = (torch/paddle-1)
