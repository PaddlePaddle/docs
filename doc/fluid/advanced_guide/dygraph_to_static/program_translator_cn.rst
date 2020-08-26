动态图转静态图

PaddlePaddle动态图模式写的代码将按照我们编写命令的顺序进行执行。这种机制更符合python程序员习惯，使得调试更加容易，并且也使得我们将大脑中的想法更轻易转化为实际代码。其具有容易debug，容易使用，灵活使用python语句的优点。不过python在部分性能上无法比过C++，工业界预测部署很多地方（如大型推荐系统，移动端）却希望直接使用C++提速，使用python的速度负担太大。这种时候静态图更具有部署和性能的优势。静态图意味着程序在编译执行时先搭建起神经网络的结构，然后再执行神经网络操作。神经网络的结构规定好后可以脱离python依赖执行。

因此动态图比静态图更容易使用，但部署性能没有静态图有优势。一种解决方法是让用户仍然使用动态图写代码，但是通过PaddlePaddle框架对用户代码的分析，转化为静态图网络结构，这就是动态图转静态图模块。这样做兼顾用户的易用性和部署性能。


基本使用方法

PaddlePaddle提供了两种动态图转静态图的方式，基于动态图trace的转换与基于源代码级别的转换。

基于trace的TracedLayer：

trace指的是在模型运行时记录下其运行过哪些算子。TracedLayer就是基于这种技术，运行一遍动态图，在动态图过程记录那些已经运行了的算子保存为静态图模型。一个使用例子如下：


```python
from paddle.imperative import TracedLayer

paddle.enable_imperative()
# 定义MNIST类的动态图模型Layer
mnist = MNIST()
in_np = np.random.random([10, 1, 28, 28]).astype('float32')
# 将numpy的ndarray类型的数据转换为Variable类型
input_var = paddle.imperative.to_variable(in_np)
# 通过 TracerLayer.trace 接口将命令式模型转换为声明式模型
out_dygraph, static_layer = TracedLayer.trace(mnist, inputs=[input_var])
save_dirname = './saved_infer_model'
# 将转换后的模型保存
static_layer.save_inference_model(save_dirname, feed=[0], fetch=[0])
```

载入的模型可以使用静态图方式运行

```python
place = paddle.CPUPlace()
exe = paddle.Executor(place)
program, feed_vars, fetch_vars = paddle.io.load_inference_model(save_dirname, exe)
fetch, = exe.run(program, feed={feed_vars[0]: in_np}, fetch_list=fetch_vars)
```


但是也正如我们阐述的原理，trace只是记录了算子，因此如果用户希望根据一些数据条件运行不同的算子，换而言之，在模型中引入依赖数据条件（包括输入的值或者shape）的控制流，则TracedLayer无法正常工作。比如下面

```python

def func(input_var)
    # if判断与输入input_var的shape有关
    if input_var.shape[0] > 1:
        out = paddle.cast(input_var, "float64")
    else:
        out = paddle.cast(input_var, "int64")

paddle.enable_imperative()
in_np = np.array([-2]).astype('int')
input_var = paddle.imperative.to_variable(in_np)
func(input_var)
```

上例如果在使用TracedLayer.trace(func, inputs=[input_var])，由于trace只能记录if-else其中跑的一次算子，模型就无法按用户想要的根据input_var的形状进行if-else控制流保存。类似的控制流还有while/for循环的情况

基于源代码转写的ProgramTranslator

对于依赖数据的控制流，我们使用基于源代码转写的ProgramTranslator来进行动态图转静态图。其基本原理是通过分析python代码来将动态图代码转写为静态图代码，并在底层自动帮用户使用执行器运行。其基本使用方法十分简便，只需要在要转化的函数（该函数也可以是用户自定义动态图Layer的forward函数）前添加一个装饰器@declarative，上面的例子转化为：

```python

from paddle.fluid.dygraph.jit import declarative

@declarative
def func(input_var)
    # if判断与输入input_var的shape有关
    if input_var.shape[0] > 1:
        out = paddle.cast(input_var, "float64")
    else:
        out = paddle.cast(input_var, "int64")

paddle.enable_imperative()
in_np = np.array([-2]).astype('int')
input_var = paddle.imperative.to_variable(in_np)
func(input_var)
```

若要存储对应的模型，可以调用ProgramTranslator单例的save_inference_model，如下例：

```python
import paddle

paddle.enable_imperative()
prog_trans = paddle.imperative.ProgramTranslator()
mnist = MNIST()

in_np = np.random.random([10, 1, 28, 28]).astype('float32')
label_np = np.random.randint(0, 10, size=(10,1)).astype( "int64")
input_var = paddle.imperative.to_variable(in_np)
label_var = paddle.imperative.to_variable(label_np)

out = mnist( input_var, label_var)

prog_trans.save_inference_model("./mnist_dy2stat", fetch=[0,1])
```

高级Debug功能

TODO：留杰，雅美的PR预计可以在2.0之前合入，其中包括打印代码，设置log，报错信息的更新。合入后进一步整理更新。

内部架构原理

TracedLayer的原理就是trace，相对简单，因此我们在这里不展开描述。本节将主要阐述ProgramTranslator基于源代码将动态图代码转化为静态图代码。


ProgramTranslator的总体架构图如下：
TODO：添加图片

我们将内部涉及的过程分为以下几步：

1. Function与缓存

2. 从Function转化为动态图源码，再进行AST（抽象语法树）解析

3. AST语法树的转写为静态图AST，再生成源码

4. 静态图源码作为动态图一部分运行的技术

5. 易用性与Debug功能在动转静过程的实现

支持的语法列表，和不支持的情况说明

