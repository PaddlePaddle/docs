# 动转静 SOT 原理及使用

## **一、背景与动机**

因为动态图和静态图在性能和用户体验、二次开发上各有优劣，深度学习框架在架构层统一动静概念并实现用户的最佳一致性使用体验，是极具挑战性的。飞桨从用户体验角度出发，着眼于训练、推理，并紧跟大模型时代的场景需求，直面技术难题，解决用户在性能，部署和大模型定制化的痛点。但是随着技术的演进，传统的基于 AST 变换的动静统一策略开始无法处理日益复杂的用户模型，在一些不常用的语法和太过于动态的场景下会出现转写失败的情况，而转写一旦失败，意味着后续基于静态图的优化工作没有任何一个可以被运用。AST 转写方案虽然具有高层级，易于转写的特性，但由于 Python 是一门纯动态语言，以及 Paddle 静态化数据表示能力的有限性，现在的 AST 方案存在如下局限性：

- **难以处理动态和静态相互混合的场景**。例如 numpy 和 tensor 的互相转换，见样例代码；

- **控制流和容器的混合使用时有边界 case**，经常出现解析出错或者是无法完全表示的情况；

- **不支持源码加密场景下使用，完备性存在上限**。比如在 C++ 端执行的 Python 代码无法进行 AST 转写，或者对于 `.pyc` 文件无法处理（某些加密场景等）


一个在套件和用户使用中经典的场景如下：

```python
# 一个简单的 Case 如下：
@paddle.jit.to_static()
def unsupport_func(x):
    x = 2 * x
    t = x.numpy() # t 依赖了 x 的值，依赖静态图的执行结果
    t = np.ones(t)
    return paddle.to_tensor(t)

x = paddle.to_tensor([2])
unsupport_func(x)  # raise error
```

这里的 np.ones 因为动转静的使用，上述的 x 和 t 其实都是 Variable 类型，传入到 np.ones 中是无法获取到真实的 value 的，因此 numpy 接口会报错。而这样的 Case 也是 AST 动转静理论上无法解决的问题，本质原因是，AST 必须要求转写的函数可以**被整图的**静态图 IR 表示。

这些长尾的 case 虽然可以通过要求用户使用规范的做法来避免，但是这类问题还是层出不穷，因为用户不希望在写动态图时考虑动转静的场景。

为了解决极具灵活性的 Python 语言与深度学习框架中间表示巨大的差异性鸿沟问题，飞桨在动转静模块中引入了字节码符号化模拟执行机制（即 Symbolic OpCode Translator，简称 SOT），在字节码层级分析和模拟执行动态图模型代码，动态抽取静态组网代码，构建成为一个新的等价的 Python 函数，消除语言与表示之间的鸿沟，实现动态图到静态图的等价转写。同时这种字节码模拟执行机制，可以自适应选择触发子图级别的打断机制（即 Graph Break），实现控制流代码保持动态图运行的效果。

## 二、概要介绍

### **自适应打断机制：**

在新的 SOT 方案中引入了自适应打断机制来获得 100% 理论动转静成功率。在旧的动转静 AST 方案中，是以源码转写的方式对整图进行转写，当遇到无法静态化的 Op 时，AST 整图转写失败。新的 SOT 方案中，首先将源码转写的方式升级为了字节码转写，当遇到无法静态化的 Op 时，我们将整图切分为子图，并使用**字节码进行粘连**，以达到转写成功的目的。在自适应打断机制加持下，用户动态图编写可以更加随意，并在子图层面享受动转静和编译器加速。

<p align="center">
    <img src="https://raw.githubusercontent.com/PaddlePaddle/docs/develop/docs/guides/paddle_v3_features/images/sot/sot_vs_ast.png" width="80%"/>
</p>


### 执行流程：

在新的 SOT 流程下，动转静是在字节码层面进行分析的，SOT 会先利用注册的 Python EvalFrame Hooker 获取到用户函数运行时的字节码和 PyFrame 上下文信息（包含了局部变量，参数等），然后使用内部实现的**字节码模拟执行器**来进行模拟执行，最后得到一个可以替换原来字节码的新 PyCodeObject 对象。模拟执行器会识别出用户函数中需要静态化的字节码和无法静态化的字节码，对于无法静态化的字节码使用打断功能会回退到动态图执行，对于可以静态化的字节码会生成一个静态图来进行替换。当第二次执行时，SOT 会先判断是否命中了上次转写的缓存，如果命中了缓存就可以直接获取上次转写的 PyCodeObject 重用。下图是整个 SOT 的执行流程。

<p align="center">
    <img src="https://raw.githubusercontent.com/PaddlePaddle/docs/develop/docs/guides/paddle_v3_features/images/sot/sot_procedure.svg" width="80%"/>
</p>

## 三、框架架构

<p align="center">
    <img src="https://raw.githubusercontent.com/PaddlePaddle/docs/develop/docs/guides/paddle_v3_features/images/sot/sot_framework.png" width="80%"/>
</p>


上图展示了 SOT 的所有组件，针对一些名词和模块，这里进行一个简单的介绍：

### **一、EvalFrame Hooker 模块**

Python 在 2016 年的 PEP523 提案支持了自定义回调函数，将默认的执行器替换为用户自定义的解释函数。这个机制结合子图 fallback 方案的需求，我们在 Paddle 的 Pybind 层暴露了 `paddle.core.set_eval_frame` 接口。

### **二、字节码模拟器（OpcodeExecutor）模块**

这个部分是 SOT 方案的核心，主要的功能是我们需要模拟获取到的 PyCodeObject，并进行动态和静态代码分离，因此字节码模拟器是将 Python 函数映射为新的 Python 函数的模块。对于不同的静态化程度的函数，**字节码模拟器**会将一个函数对应于下面几种可能的情况：

1. 若能够**完全静态化**目标函数，则需要返回一个新的可执行函数，该函数能够构建目标函数对应的子图；
2. 若只能**部分静态化**目标函数，同样需要返回一个新的可执行函数，该函数将可静态化部分抽取为子图，并将无法静态化的部分抽取为子函数（可能代表着不同分支），通过 Eval Frame 机制进行递归的处理。
3. 若完全**无法静态化**目标函数，则返回原本的目标函数，在动态图环境下进行计算。

我们在 SOT 项目中完成了一个完备的 Python 字节码解释器，具有如下的特点：

- 设计良好，具备 Dispatch 机制，符合开闭原则，便于维护。
- 支持随意触发打断和 Fallback 的能力。
- 支持子函数递归模拟。
- 完备的字节码支持，完备的版本支持。我们支持 python3.8 - python3.12 的几乎 90%常见字节码模拟。

### **三、自适应子图打断模块：**

**对于控制流 If、For 依赖 Tensor 的场景，需要打断构图并静态化部分函数，子图打断能力是 SOT 能够达到近 100%成功率的核心组件。**

我们深入研究了打断的类型，设计和打断机制，并将所有的打断场景划分为了 2 个不同的行为：

- BreakGraph ：触发子图打断，当前函数会产生一个子图和一个 resume function 进行下一轮的模拟。
- Fallback：触发子图打断，当前函数不产生子图，直接动态图运行。

基于不同的场景我们设计了不同的异常传播途径和不同的处理逻辑。

### **四、Tracker、Guard、缓存模块：**

子图 Fallback 的整体实现可以认为是将用户函数原始字节码转换为新的字节码，**为了避免每次传入相同输入都会重新触发开销昂贵的字节码转换操作，我们需要增加缓存机制来复用之前转写过的代码，实现 JIT 的效果。**

但并不是任何字节码成功转换一次后第二次都是可以直接复用的，因为我们字节码的转换是基于 Frame 的初始状态进行模拟执行得到的，也就是说**转换后的字节码强依赖于 Frame 的初始状态**。当初始状态发生改变，最后转换后的字节码很有可能发生改变，因此我们需要一种机制来根据 Frame 初始状态来判断缓存过的字节码是否有效。这种转换复用的机制我们称为 Guard 函数，而 Guard 函数生成依赖字节码模拟过程中记录的每个模拟变量的 Tracker。

### **五、副作用处理模块：**

**SideEffect 是指代码执行过程中除了函数返回值之外，还对调用方产生了额外的影响，比如修改全局变量、修改可变的共享变量等。**

在模拟执行过程中，我们的代码是在虚拟环境下执行的，在该过程中不应该也不会对真实环境进行修改。而如果用户代码产生了 SideEffect，我们需要在生成的代码里反映出相应的 SideEffect，即在字节码生成步骤中增加 SideEffect 的处理部分。副作用模块就是专门记录并处理副作用正确性的功能模块。

### **六、StatementIR 模块：**

**StatementIR 是 Paddle 动转静模块与子图 FallBack 的一个『中间桥梁』，它达到了动转静复用的目的。**

StatementIR 与 Program 类似，都是表征计算的一个结构。**在字节码执行过程中，我们需要将所有的组网代码都『临时记录』下来，并最后将他们组网成为一个 Program 。**这里的组网代码记录的载体就是 StatementIR 。在函数结束的时刻，我们会将记录下来的 StatementIR 转化为一个函数。与原来的用户代码不同，由 StatementIR 转化为的函数可以确保一定可以动转静。这样我们可以复用原来的动转静 to_static 函数来实现静态图的执行。

## 四、对比 AST 方案

SOT 方案相比于 AST 方案有如下的优势：

1. 【成功率提升】SOT 在遇到不支持的语法时会自动打断，并将不支持部分运行在动态图下，因此理论上可以达到近 100% 的成功率。
2. 【转写完备性】SOT 只依赖 Python 字节码，针对无法获取源码的场景，也可以得到运行，获取正确的结果。
3. 【控制流支持】SOT 因为支持自适应子图打断，因此可以不静态图化某些容器操作，可以更好的处理控制流与容器。不需要在静态图底层支持太多的容器类结构，比如 TensorArray 或者是 TensorDict。
4. 【自适应打断子图】SOT 支持自适应打断子图。在无法静态化时，主动打断组网、运行静态图并获取输出，然后在进行新一轮的组网。因此可以在自图层面享受静态图和编译器的加速收益。

**注意：在 Save/Load 模式下需要整图导出，会自动切换到 AST 模式进行运行。**

## 五、开始使用

### 使用 SOT 模式（默认模式）

目前 SOT 模式是动转静的默认转写模式。用户只需要使用默认的 paddle.jit.to_static 就可以，下面是一个 SOT 动转静的使用样例：

```python
import paddle
from paddle.jit import to_static
from paddle.static import InputSpec
import numpy as np
import random

# set seed for determinated output
paddle.seed(2024)
np.random.seed(2024)
random.seed(2024)

class SimpleNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.linear = paddle.nn.Linear(10, 3)

    def forward(self, x, y):
        x = self.linear(x)
        x = x + y
        np_x = x.numpy()
        np_x = np.sum(np_x) * 2
        return paddle.to_tensor(np_x)

net = SimpleNet()

net = paddle.jit.to_static(net, full_graph=False)  # 动静转换, full_graph=False 表示 SOT 模式
x = paddle.randn((10, 10))
y = paddle.randn((3,))
out = net(x, y)
print(out)
```

输出如下：

```bash
Tensor(shape=[], dtype=float64, place=Place(gpu:0), stop_gradient=True,
       54.16428375)
```

### 使用 AST 模式

如果确定自己的代码完全可以静态化，用户可以手动打开 AST 模式，通常 AST 模式成功率会更低，但是调度开销会更小，同时支持部署推理。

```python
import paddle
from paddle.jit import to_static
from paddle.static import InputSpec
import numpy as np
import random

# set seed for determinated output
paddle.seed(2024)
np.random.seed(2024)
random.seed(2024)

class SimpleNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.linear = paddle.nn.Linear(10, 3)

    def forward(self, x, y):
        x = self.linear(x)
        x = x + y
        np_x = x.numpy()
        np_x = np.sum(np_x) * 2
        return paddle.to_tensor(np_x)

net = SimpleNet()

net = paddle.jit.to_static(net, full_graph=True)  # 动静转换, full_graph=True 表示 AST 模式
x = paddle.randn((10, 10))
y = paddle.randn((3,))
out = net(x, y)
print(out)
```

AST 模式下会报错，因为我们尝试混合使用 numpy 和 paddle api，导致无法整图静态化。
```bash
Traceback (most recent call last):
  File "ttt.py", line 29, in <module>
    out = net(x, y)
  File "/home/ssd2/xiongkun/Paddle/build/python/paddle/nn/layer/layers.py", line 1484, in __call__
    return self.forward(*inputs, **kwargs)
  File "/home/ssd2/xiongkun/Paddle/build/python/paddle/jit/dy2static/program_translator.py", line 502, in __call__
    return self._perform_call(*args, **kwargs)
  File "/home/ssd2/xiongkun/Paddle/build/python/paddle/jit/dy2static/program_translator.py", line 822, in _perform_call
    error_data.raise_new_exception()
  File "/home/ssd2/xiongkun/Paddle/build/python/paddle/jit/dy2static/error.py", line 448, in raise_new_exception
    raise new_exception from None
TypeError: In transformed code:

    File "ttt.py", line 21, in forward
        x = x + y
        np_x = x.numpy()
        np_x = np.sum(np_x) * 2
        ~~~~~~~~~~~~~~~~~~~~~~~ <--- HERE
        return paddle.to_tensor(np_x)

    File "<__array_function__ internals>", line 200, in sum

    File "/root/miniconda3/envs/py38/lib/python3.8/site-packages/numpy/core/fromnumeric.py", line 2324, in sum
        return _wrapreduction(a, np.add, 'sum', axis, dtype, out, keepdims=keepdims,
    File "/root/miniconda3/envs/py38/lib/python3.8/site-packages/numpy/core/fromnumeric.py", line 84, in _wrapreduction
        return reduction(axis=axis, out=out, **passkwargs)

    TypeError: Code 'np_x = np.sum(np_x) * 2' called numpy API np.sum, please use Paddle API to replace it.
               values will be changed to variables by dy2static, numpy api can not handle variables

```
