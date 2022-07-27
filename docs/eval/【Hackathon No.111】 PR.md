- 一个完整的使用动静转换@to_static 导出、可部署的模型完整代码（参考以图搜图），提供 AI Studio 任务链接
AI Studio 任务链接:https://aistudio.baidu.com/aistudio/projectdetail/3910079

- 接口层面：

接口层面相对来说比较全面，指出了模型静态图导出的方法。同时 InputSpec 也比较好用，可以通过三种方式来构造所需要的 InputSpec：直接构造、由 Tensor 构造以及由 numpy.ndarray 构造，但是并没有指出这三种方式构造的 InputSpec 的优缺点。在动态图转静态图--使用样例--2.2.2 基本用法的方式四：指定非 Tensor 参数类型中代码有问题，to_static 函数中没有输入 net 参数，修改代码如下：

```python
class SimpleNet(Layer):
    def __init__(self, ):
        super(SimpleNet, self).__init__()
        self.linear = paddle.nn.Linear(10, 3)
        self.relu = paddle.nn.ReLU()

    def forward(self, x, use_act=False):
        out = self.linear(x)
        if use_act:
            out = self.relu(out)
        return out

net = SimpleNet()
# 方式一：save inference model with use_act=False
# 修改
net = to_static(net, input_spec=[InputSpec(shape=[None, 10], name='x')])
paddle.jit.save(net, path='./simple_net')


# 方式二：save inference model with use_act=True
# 修改
net = to_static(net, input_spec=[InputSpec(shape=[None, 10], name='x'), True])
paddle.jit.save(net, path='./simple_net')
```
- 语法层面：

支持语法相对来说是比较全面的，介绍的也比较细致。控制流语法等用起来也比较流畅。但是在第三方相关库 numpy 中只是简单的说了部分支持，并没有具体的例子解释 numpy 操作中哪部分是支持的，哪部分是不支持的。并且在案例解析--三、内嵌 Numpy 操作中直接写到动态图模型代码中 numpy 相关的操作不能转换为静态图，虽然提供了一个好的方法来解决这个问题（转换为 tensor），虽然能理解下来但是感觉这两部分写的不具体且有点矛盾。
![4df67d8440d0fc20490cbd09cbd5498](https://user-images.githubusercontent.com/102226413/165878773-640e73c2-d343-4fb2-8d6b-af3947d9c6bb.png)
![5c43735dfac00b3290cf2b0b5c58b3d](https://user-images.githubusercontent.com/102226413/165878786-ed404b8c-ab03-43a7-9b15-9dc56dc44635.png)


另外，在案例解析 6.1 默认参数的部分，给出了 forward 函数一个不错的建议，但是当我在分析它的原因的时候，我测试了一下下面的代码：


```python
import paddle
from paddle.nn import Layer
from paddle.jit import to_static
from paddle.static import InputSpec


class SimpleNet(Layer):
    def __init__(self, ):
        super(SimpleNet, self).__init__()
        self.linear = paddle.nn.Linear(10, 3)
        self.relu = paddle.nn.ReLU()

    def forward(self, x, use_act=False):
        out = self.linear(x)
        if use_act:
            out = self.relu(out)
        return out

net = SimpleNet()
# 方式一：save inference model with use_act=False
# paddle.jit.save(net, path='./simple_net', input_spec=[InputSpec(shape=[None, 10], name='x')])


# 方式二：save inference model with use_act=True
paddle.jit.save(net, path='./simple_net', input_spec=[InputSpec(shape=[None, 10], name='x'), True])

```
它他并没有报错，但是 paddle.jit.save 时在 input_spec 时我指定了非 tensor 的数据，而且程序运行并没有报错，这会不会与原因有点冲突？文档原因截图如下：
![f4e2808997a5556bfd5f6c580245b3f](https://user-images.githubusercontent.com/102226413/165878738-61ed378a-67cb-4d0e-93b8-aba8e7b6fe13.png)



- 报错层面

文档总体来说写的比较全面。
文档中 1.1 错误日志怎么看，报错调试的文档代码如下：
```python
import paddle
import numpy as np

@paddle.jit.to_static
def func(x):
    two = paddle.full(shape=[1], fill_value=2, dtype="int32")
    x = paddle.reshape(x, shape=[1, two])
    return x

def train():
    x = paddle.to_tensor(np.ones([3]).astype("int32"))
    func(x)

if __name__ == '__main__':
    train()

```
报错日志如下图，在 paddle 内置的方法中有点难以快速定位到问题所在。该报错问题应该是第 7 行 paddle.reshape 的维度设置不对。但是在使用排错日志的时候，没有报错信息直接定位到第 7 行。个人觉得对错误代码位置的直接定位才是最重要的。而且报错的内容提示太多，对新手来说不会很友好。建议直接在报错的时候，报错的最后位置，重复一遍，最重要的报错信息，并提示报错代码所在位置。这样对新手比较友好。对于这种简单问题的报错提示更加明确一点会让使用者觉得更加方便。
![@6U(A`{~$P$`XD1I{YGYOLT](https://user-images.githubusercontent.com/102226413/165878813-ec7a90b6-518b-4a2c-ae68-8a92572ff96a.png)
![)PH6WOHJZ{UJ}~YIKADQ)$4](https://user-images.githubusercontent.com/102226413/165878824-4d3dfe4f-3dea-447d-86fe-5d57c0937246.png)



- 文档层面

文档整体比较完善，但是在使用指南->动态图转静态图->案例解析 中全部都是动静转化机制的各种 API 的分章节介绍，建议在案例解析最后增加一个完整的实例代码，比如 cifar10 图像分类的动态图转静态图案例，或者把应用实践中的案例链接附在最后，方便读者找寻。有些读者可能想找一个案例，然后找了使用指南的案例解析，发现没有一个完整的案例，正巧这个读者对整个文档不熟悉，没看过应用实践，然后就找不到案例。


- 意见建议（问题汇总）

1、接口层面，使用指南->动态图转静态图->使用样例 2.2.1 构造 inputSpec 并没有指出这三种方式构造的 InputSpec 的优缺点。
2、语法层面，对 numpy 的支持性存在一些问题。
3、报错调试，在使用排错日志的时候，没有报错信息直接定位到错误代码的位置，且报错内容提示太多，对新手不友好。建议直接在报错的时候，报错的最后位置，重复一遍，最重要的报错信息，并提示报错代码所在位置。
4、文档层面，在使用指南->动态图转静态图->案例解析 中全部都是动静转化机制的各种 API 的分章节介绍，建议在案例解析最后增加一个完整的实例代码。
