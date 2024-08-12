# Limitations


飞桨动转静（[@to_static](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/jit/basic_usage_cn.html)）目前已支持大多数 [Python 语法](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/jit/grammar_list_cn.html)，实现了动态图模型一键转为静态图训练和部署。但因 Python 语法极大的灵活性，飞桨动转静在某些场景下尚存在一定的局限性，需要用户按照一定的规范编写模型代码，以提升转写成功率。

本文档将结合具体的代码样例，对飞桨动转静的局限性（即 Limitations）进行阐释，并给出规范性代码推荐写法。若在使用动转静遇到了类似问题，可查阅此文档中的指南和建议，让模型动转静更加的流畅，主要包括如下几个场景：

1. **控制流**：主要涉及动态图代码中存在 [if...else](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/jit/principle_cn.html#ifelse) 和 [for/while](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/jit/principle_cn.html#for-while) 的场景；
2. **容器类**：主要涉及动态图代码中搭配控制流使用 Python 容器的场景；
3. **语法类**：主要涉及动态图代码中包含动转静尚不支持语法的场景；

## 一、控制流
### 1. if...else 语句

#### 1.1 变量在不同分支类型须保持一致

模型代码中的 if...else 语句在动转静之后，会被[转换成统一的范式](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/jit/principle_cn.html#ifelse)。当依赖的条件变量（如下样例中的 `x > y` ）是一个 Tensor 类型时，if...else 分支中所有的变量类型须保持一致。当类型不一致时，后续的类型检查将抛出异常。

如下是一个典型的代码样例：

```python
import paddle
from paddle.jit import to_static

@to_static
def func(x, y):
    if x > y:
        y = paddle.to_tensor(3)  # <--- b 是 Tensor 类型
    else:
        y = True                 # <--- b 是内建 bool 类型

    if y == True:                # <--- 对 y 进行判断，将在动转静时引发错误
        x = x + 1
    return x, y

x = paddle.to_tensor(1)
y = paddle.to_tensor(2)
out = func(x, y)
```

上述代码将报如下错误：`InvalidArgumentError: The type of data we are trying to retrieve does not match the type of data currently contained in the container.`

**规范性写法**：对不同的类型可使用额外的变量，将功能进行拆分。

```python
import paddle
from paddle.jit import to_static

@to_static
def func(x, y):
    if x > y:
        y = paddle.to_tensor(3)    # <--- y 始终是 Tensor 类型
        flag = False               # <--- flag 始终是内建 bool 类型
    else:
        flag = True

    if flag == True:
        x = x + 1
    return x, y

x = paddle.to_tensor(1)
y = paddle.to_tensor(2)
out = func(x, y)
```


#### 1.2 张量在不同分支 shape 须保持一致

依赖控制流的 if...else 语句在动转静生成中间表示 Program 时，要求两个分支中同名张量的 shape 必须保持一致，因为静态图下会对两个分支的输出进行动态 `select input` 操作，故须保证无论条件变量 `x > y` 取何值，选取的张量 shape 都是一致的。否则在后续组网或者训练时，出现因 shape 不同而报错。

如下是一个典型的代码样例：

```python
import paddle
from paddle.jit import to_static

@to_static
def fun(x, y):
    z = paddle.randn([2, 3])

    if x > y:
        y = paddle.randn([2, 2])          # <--- y.shape 是[2, 2]
    else:
        y = paddle.randn([4, 5])          # <--- y.shape 是[4, 5]

    out = paddle.concat([y, z], axis=-1)  # <--- y 与 z 不能保证始终能 concat 成功
    return out

x = paddle.to_tensor(1)
y = paddle.to_tensor(2)
out = fun(x, y)
```

上述代码将报如下错误：`InvalidArgumentError: The 0-th dimension of input[0] and input[1] is expected to be equal.But received input[0]'s shape = [4, 5], input[1]'s shape = [2, 3].` 。

**规范性写法**：调整依赖控制流的 if...else 不同分支同名张量的代码逻辑，确保 shape 保持一致

### 2. for、while 语句

#### 2.1 条件变量类型须保持不变

While 的条件变量在循环过程中的类型应保持不变，因为[循环变量的类型](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/jit/principle_cn.html#for-while)将会决定其是保持 Python 语法运行，或是转为飞桨的 [while_loop](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/static/nn/while_loop_cn.html#while-loop) API。保持条件变量类型不变才能确保模型正确地被动转静。

如下是一个典型的代码样例：

```python
import paddle
from paddle.jit import to_static

@to_static
def func(x : paddle.Tensor):
    t = 2
    while t < 10:               # <--- 初始为 bool 类型，循环一次后为 Tensor 类型
        t = paddle.shape(x)[0]  # <--- t 变为了 Tensor 类型
        x = paddle.concat([x, x])
    return x

x = paddle.randn([2, 3])
out = func(x)
```
如上述样例在执行循环体后，条件变量 `t < 10` 从 Python 的 bool 类型变为了 Tensor 类型，动转静报错机制会捕捉并抛出异常：`Dygraph2StaticException: python while pred change from bool to Tensor. ` 。

此处根据期望模型代码运行的效果，**有如下两种「规范性」的写法：**

* 若此处是一个循环次数固定的 while，则应避免 `t` 的类型变化，规范性写法为：

```python
def func(x : paddle.Tensor):
    t = 2
    while t < 10:
        t = x.shape[0]   # <--- 借助 x.shape 获取 int 类型值
        x = paddle.concat([x, x])
    return x
```

+ 若此处是一个循环次数不固定的 while，则可以将 `t` 的类型提前转为 Tensor，规范性写法为：

```python
def func(x : paddle.Tensor):
    t = paddle.to_tensor(2)   # <--- 提前转为 Tensor 类型
    while t >= 0:
        t = paddle.shape(x)[0]
        x = paddle.concat([x, x])
    return x
```
对于 `for i in y` 的情况，可以调整为 `for i in paddle.to_tensor(y)` 即可。

#### 2.2 变量类型结构须保持一致

动态图模型中的 for、while 语句在动转静时，当其属于[依赖控制流的情况](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/jit/principle_cn.html#for-while)时，其循环体内代码块逻辑只会被执行一次。若循环前后的同名变量的数据类型、数据结构不一致，将无法保证生成的中间表示 Program 涵盖动态图下所有逻辑，出现转写不等价的情况。

常见的错误有循环前为 None，循环后为 Tensor。这类情况建议使用默认值方法或者是 do-while 形式替换。

如下是一个典型的代码样例：

```python
import paddle
from paddle.jit import to_static

@to_static
def func(x : paddle.Tensor):
    cache = None               # <--- cache 循环前为 None
    y = paddle.to_tensor(0.)
    while y.mean() < 10:
        if cache is None:
            y = x + 1
        else:
            y = x + cache
        cache = y              # <--- cache 循环后为 Tensor

    return y

x = paddle.to_tensor(1.)
out = func(x)                  # <--- 动转静执行结果与动态图不一致
```

上面的例子中，`cache` 在第一轮循环中为 None，在后续的轮次中为 Tensor，而目前无法同时表示 None 与 Tensor, 导致动转静后执行结果错误。

**规范性写法**：可以将循环代码修改为如下的 do-while 形式，确保循环前后的数据类型一致性


```python
import paddle
from paddle.jit import to_static

@to_static
def func(x : paddle.Tensor):
    y = x + 1                  # <--- 调整为 do-while 形式
    cache = y                  # <--- cache 类型始终为 Tensor
    while y.mean() < 10:
        y = x + cache
        cache = y

    return y

x = paddle.to_tensor(1.)
out = func(x)                  # <--- 动转静执行结果与动态图一致
```

#### 2.3 迭代变量 Shape 须保持不变

while 循环体中迭代变量的值可以变化，但是其 shape 须保持不变，否则可能导致隐式的错误（精度对不齐、结果错误之类）。若无法避免，可以通过动态 shape 来解决。

如下是一个典型的代码样例：

```python
import paddle
from paddle.jit import to_static

@to_static
def func(x, y):
    for i in range(paddle.to_tensor(3)):
        x = paddle.concat([x, y], axis=0)

    print(x.shape)    # <--- 动态图下返回 [8, 3], 静态图下返回[4, 3]
    # .....           # <--- 若此处存在依赖 x.shape[0] 的代码逻辑，存在隐式错误
    return x

x = paddle.randn([2, 3])
y = paddle.randn([2, 3])
out = func(x, y)
print(out.shape)
```

上述代码可以正确转写，但是有风险。转写成功之后 `x` 的编译期 shape 会与执行期的 shape 有所差别，可能会影响后续的组网。若模型中存在类似 `x = paddle.concat([x, y], axis=0)` 之类的对 `x.shape` 进行改写的操作，建议提前给 `x` 的 shape 全部变为 -1，以防止组网错误。

```python
import paddle
from paddle.jit import to_static

@to_static
def func(x, y):
    x = paddle.reshape(x, paddle.shape(x)) # <--- 将 x.shape 变为了(-1, -1, -1)可以防止组网错误
    for i in range(paddle.to_tensor(3)):
        x = paddle.concat([x, y], axis=0)
    return x

x = paddle.randn([2, 3])
y = paddle.randn([2, 3])
out = func(x, y)
print(out.shape)
```

## 二、容器类

容器类与控制流关系密切，目前支持情况如下：

|              | **依赖 Tensor 控制流** | **不依赖 Tensor 控制流** |
|:------------:|:--------------------:|:----------------------:|
| **有变长操作** | 仅支持无嵌套的 list 且 Value 必须可 Tensor 化 | 支持所有场景 |
| **无变长操作** | 支持元素修改，但 Value 必须可 Tensor 化 | 支持所有场景 |

1. **变长操作**，对 list 而言包含：append、push、del 等会改变容器结构的操作；对于 dict 而言，还包含插入一个之前未包含的 key（进入控制流和退出控制流结构对比）
2. **可 Tensor 化**，表示当前的 Tensor 存在对应的静态图结构，比如 int、float、double、bool 等会转化为 Tensor；用户自定义的类无法 Tensor 化

如果模型代码中容器的结构本身会发生变化，动转静时会将其对应的 list 转写为[静态图的 TensorArray 数据结构](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.4rc/guides/jit/case_analysis_cn.html#list-lodtensorarray)。

### 1. 有变长操作

#### 1.1 不支持多层 list 嵌套

在依赖 Tensor 的控制流中，涉及 append、pop 操作的 list 会被转为 TensorArray。由于目前飞桨框架的 TensorArray 仅能表示单层语义，故当前的动转静不支持对嵌套的 list 进行变长处理。

如下是一个暂未支持的使用样例：

```python
import paddle
from paddle.jit import to_static

@to_static
def func(x):
    t = paddle.shape(x)[0]
    out = [[1,2], [2, 3]]  # <--- out 是嵌套的 list
    for i in range(t):     # <--- 依赖 Tensor 的控制流
        out.append(i)      # <--- 变长的 list，会触发转写为 tensor array，导出报错：不支持嵌套

    return out

x = paddle.randn([2, 3])
out = func(x)
print(out) # 动态图下为[[1, 2], [2, 3], 0, 1]，静态图下报错
```

动转静会报错，报错信息如下：
```python
TypeError: In transformed code:

    File "test.py", line 5, in func
        t = paddle.shape(x[0])
        out = [[1,2], [2,3]]
        for i in range(t):
        ~~~~~~~~~~~~~~~~~~ <--- HERE
            out.append(i)
        return out

TypeError: All values in `initialized_list` should be Variable, but recevied <class 'list'>.
```

**修改建议**：推荐将 [1,2] 和 [2,3] 转为 tensor，变为单层 list。如下面写法：

```python
import paddle
from paddle.jit import to_static

@to_static
def func(x):
    t = paddle.shape(x)[0]
    out = [paddle.to_tensor([1, 2]), paddle.to_tensor([2, 3])] # <--- 变为单层 list
    for i in range(t):
        out.append(i)

    return out

x = paddle.randn([2, 3])
out = func(x)
print(out)
```

> 注：对于类型不同，导致无法转成同一类型 Tensor 的情况，比如[1, "NCHW"]。建议不要将此类对象放入到依赖 Tensor 的控制流中。

#### 1.2 仅支持 list 有限操作
在依赖控制流的场景下，目前动转静仅支持 list 的高频用法，建议使用如下的接口来操作 list：

1. 支持 append、pop，比如 `x.append(0 / tensor)`
2. 支持 getitem，比如 `x[0]`
3. 支持 int、Tensor 作为 index 的 setitem ，比如 `x[0] = 1`
4. 支持 slice， 比如 `x[0:3]`
5. 暂不支持 del 内部元素，比如 `del x[0]`
6. 暂不支持 slice 作为 index 的 setitem，比如 `x[0:2] = 1`

**一个完整的例子如下：**

```python
import paddle
from paddle.jit import to_static

@to_static
def func(x):
    res = []
    for i in range(x):  # <--- 依赖 Tensor 的控制流
        res.append(1)   # <--- 支持。因为 res 隐式转换为 TensorArray
        res.append(x)   # <--- 支持
        res.pop(-1)     # <--- 支持。删除最后一个元素
        res[0] = 12     # <--- 支持。覆写第 0 个元素为 12
        # del res[0]    # <--- 不支持。建议使用 a.pop(0) 替代
        # res[0:1] = 12 # <--- 不支持。不支持 slice 作为 setitem 的索引，只支持简单的 int / Tensor 索引，请使用 pop 等来进行组合操作达到目的。

    out = 0.
    for i in res:       # <--- 支持。无嵌套的 list 支持迭代，可以依次取到所有的值，for 变为了依赖 Tensor 的控制流
        out += i
    return out          # <--- 最后的 s 是所有 a 中的元素的 sum

x = paddle.to_tensor(3)
out = func(x)
print(out) # 返回值为 14.0
```

从上述代码样例可以看出：

+ **为控制流场景**。for 是一个依赖 Tensor 的控制流；
+ **必须满足非嵌套 list**。如变量 `res` ；
+ **支持高频 list 操作**。如只支持：赋值、append、pop 操作，不支持 del 等操作，有其他复杂操作请组合上述有限操作来实现；

#### 1.3 有限支持 dict 等其他容器

区别于控制流中 list 容器，目前飞桨底层并没有提供类似 TensorDict 数据结构，所以尽量避免在依赖 Tensor 的控制流中使用结构有变化的 dict 容器。

如下是一个暂未支持的样例：

```python
import paddle
from paddle.jit import to_static

@to_static
def func(x):
    res = { 'a': 1 }
    t = paddle.shape(x)[0]
    for i in range(t):      # <--- 依赖 Tensor 的控制流
        res['b'] = i        # <--- 不支持。因为在一个依赖 Tensor 的控制流中修改了 dict 结构
    return res

x = paddle.randn([2, 3])
out = func(x)
print(out)
```

上述代码在动转静时会报错：`ValueError: var range_0.tmp_0_slice_0 not in this block。因为变量 res 在 for 循环之前的 keys 是 {'a'}，而 for 循环之后是 {'a', 'b'}.` 。

**修改建议：**

+ 先判断是否可以消除控制流对 Tensor 的依赖，调整为不依赖 Tensor 的控制流，则 dict 等容器的操作都是支持的
+ 可以让 dict 的结构（keys）在进入控制流之前固定，且不在控制流中进行 keys 的增删操作

对于修改方法 1，规范性代码写法为：

```python
def func(x):
    res = { 'a': 1 }
    t = x.shape[0]
    for i in range(t): # <--- 不依赖 Tensor 的控制流，即 Python 控制流
        res['b'] = i   # <--- 支持
    return res
```

对于修改方法 2，规范性代码写法为：

```python
def func(x):
    res = { 'a': 1, 'b': -1 } # <--- 使用一个占位符，提前占位
    t = paddle.shape(x)[0]
    for i in range(t):        # <--- 依赖 Tensor 的控制流
        res['b'] = i          # <--- 支持。for 循环前后 a 的 value 变化了，但是 a 的结构 （keys）没有发生变化，都是{'a', 'b'}
    return res
```

### 2. 无变长操作

#### 2.1 给容器添加赋值语义

主要针对依赖 Tensor 的控制流场景，确保代码中存在显式地对容器的赋值语义。如下代码，我们在依赖 Tensor 的控制流中修改了变量 `res` 的值，必须给与一个赋值语义才能保证结果的正确性。所以按照第二段代码修改添加类似 `res = res` 赋值语句即可。

```python
def func(x):
    re = { 'a': 1 }
    t = paddle.shape(x)[0]
    for i in range(t):        # <--- 依赖 Tensor 的控制流
        a['a'] = i            # <--- 不支持，修改了 res 的元素，如果想要正确转换成功，比如给 res 添加一个赋值语义。
    return a
```

**规范性写法：**
```python
def func(x):
    re = { 'a': 1 }
    t = paddle.shape(x)[0]
    for i in range(t):        # <--- 依赖 Tensor 的控制流
        res = res             # <--- 给 res 容器一个赋值语义，表明在 for 中修改了 res 的元素。
        a['a'] = i
    return a
```

#### 2.2 分离可变与常量字段

在动转静过程中，当一个 dict 的元素参与组网，会隐式地将所有变量转换为静态图 Tensor 类型，若后续代码中使用了字典中的常量数据进行判断，可能导致错误。建议将容器的可变数据和与常量数据进行分离，分别存储在两个分开的容器中，可以提升动转静成功率。

如下是 GPT 模型中一个样例：

```python
# 错误例子
def generation(self, input_ids, ...):
    model_kwargs = {'a': 1, 'use_cache': True} # <--- 字典中的 a 参与组网。而 use_cache 是一个用户设置，不参与组网
    while flag_tensor:
        outs = self.forward(model_kwargs)
        model_kwargs = update_kwargs(model_kwargs, outs)  # <--- 利用 forward 的结果更新字典中的 a，由此触发了整个字典内容的转换
        if model_kwargs['use_cache'] is True:  # <--- use_cache 被转写之后不再是 true， 而是 Tensor 类型，导致下面的判断结果为 False
            pass
```
**规范性写法**：可根据 keys 对应的 value 是否可变，将 dict 拆分为两部分。对于不会变化的常量数据，可以单独定义或放到 `class.__init__` 提前定义。

```python
# 修改例子
def generation(self, input_ids, ...):
    model_immutable = {'use_cache': True}  # <--- dict 中常量字段
    model_mutable = {'a': 1}               # <--- dict 中可变字段
    while flag_tensor:
        out = self.forward(dict(**model_mutable, **model_immutable))
        model_mutable = update_kwargs(model_mutable, out) # <--- 仅 update 记录了可变数据的字典，即可避免不必要的转换
        if model_kwargs['use_cache'] == True:
            pass
```

## 三、语法类

### 1. 使用 paddle 代替 numpy 接口

动转静组网时，所有的张量都被转换为静态图 Tensor 类型，其在组网编译期是没有数据区的。故此时 numpy API 无法对静态图 Tensor 进行数值计算，建议将使用 paddle API 进行替换，以生成正确的中间表示 Program。

如下是一个典型的使用样例：

```python
import paddle
import numpy as np
from paddle.jit import to_static

# 错误例子
@to_static
def func(x):
    out = np.sum(x.numpy())  # <--- numpy 操作无法记录到 Program 中
    return out

x = paddle.to_tensor(3)
out = func(x)
print(out)
```

**规范性写法**：使用等价的 paddle API 代替 numpy 的 API

```python
import paddle
import numpy as np
from paddle.jit import to_static

@to_static
def func(x):
    out = paddle.sum(x)  # <--- 替换为 paddle.sum
    return out

x = paddle.to_tensor(3)
out = func(x)
print(out)
```

### 2. 不支持 set_state_dict 转写

目前暂不支持在被 `@to_static` 装饰的函数中调用 [set_state_dict()](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.0/api/paddle/fluid/dygraph/layers/Layer_cn.html#set_state_dict) 函数，因为这意味着动态地改变网络结构。

如下是一个典型的使用样例：

```python
import paddle
from paddle.jit import to_static

class mylayer(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.linear = paddle.nn.Linear(5, 2)

    @to_static
    def forward(self, x):
        old_dict = self.state_dict()  # <--- 动态调用 set_state_dict 函数将导致错误
        wgt = old_dict['linear.weight']
        drop_w = paddle.nn.functional.dropout(wgt)
        old_dict['linear.weight'] = drop_w
        self.set_state_dict(old_dict)

        return self.linear(x)
```

**规范性写法**：避免被 @to_static 装饰的函数中调用 [set_state_dict()](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.0/api/paddle/fluid/dygraph/layers/Layer_cn.html#set_state_dict) 函数，将它移动到 `__init__` 函数中。

```python
import paddle
from paddle.jit import to_static

class mylayer(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.linear = paddle.nn.Linear(5, 2)

        old_dict = self.state_dict()   # <--- 如果代码只需要执行一次，可以放在 __init__ 中
        wgt = old_dict['linear.weight']
        drop_w = paddle.nn.functional.dropout(wgt)
        old_dict['linear.weight'] = drop_w
        self.set_state_dict(old_dict)

    def forward(self, x):
        return self.linear(x)
```

### 3. 检查 isinstance 的使用

在动转静中，组网相关的变量有可能被转换为静态图 Tensor，因此使用 isinstance 对变量类型进行判断存在一定风险，请留意下列变量有可能转化为 Tensor：

+ np.array &rarr; Tensor
+ 非嵌套 list &rarr; TensorArray
+ int / float / double / bool &rarr; Tensor

> 注：上述的转写不是一定会发生， 动转静只会在必要时进行转写。

如下是一个典型的使用样例：

```python
import paddle
from paddle.jit import to_static

@to_static
def func(x):
    a = paddle.to_tensor(2)
    b = None
    if a > 1:    # <--- 依赖 Tensor 的控制流
        b = 1    # <--- b 被隐式转为 Tensor
    else:
        b = -1   # <--- b 被隐式转为 Tensor

    # 动转静后，b 将是一个 Tensor 而非 int，可能导致错误
    if isinstance(b, int):
        print ("b is int")
    else :
        print ("b is not int")

    return b

x = paddle.to_tensor([3])
out = func(x)
```

因此若在[调试时](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/jit/debugging_cn.html#ertiaoshifangfa)发现某个变量因为动转静转写为 Tensor 而导致了错误，可以通过修改 isinstance 语句来解决。
