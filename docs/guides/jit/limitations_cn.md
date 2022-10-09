# Limitations


飞桨动转静（@to_static）目前已支持大多数Python语法，实现动态图模型一键转为静态图训练和部署。但由于Python语法的灵活性，飞桨动转静在某些场景下存在一定的局限性，需要用户按照一定的规范和准则编写模型代码。

本文档将通过具体的代码样例对飞桨动转静的局限性（即 Limitation）进行阐释，并给出规范性代码写法。若在使用动转静遇到了类似问题，可查阅此文档中的指南和建议，可以让动转静过程更加的高效。主要包括如下几个场景：

1. **控制流**：主要涉及动态图代码中存在 if...else 和 for/while 的场景；
2. **容器类**：主要涉及动态图代码中搭配控制流使用Python容器的场景；
3. **语法类**：主要涉及动态图代码中包含动转静尚不支持语法的场景；

## 一、控制流

### 1. if...else 语句

#### 1.1 变量在不同分支类型须保持一致

模型代码中的 if...else 语句在动转静之后，会被转换成统一的范式。当依赖的条件变量（如下样例中的x > y）是一个 Tensor 类型时，if...else 分支中所有的变量类型须保持一致。当类型不一致时，后续的类型检查将抛出异常。

如下是一个错误的代码样例：

```python
import paddle
from paddle.jit import to_static

# 错误例子
@to_static
def func(x, y):
    if x > y:
        y = paddle.to_tensor(3)  # <--- b 是 Tensor 类型
    else:
        y = True                 # <--- b 是内建bool类型
    
    if y == True:   # 此处对 y 进行判断，将在动转静时引发错误
	    x = x + 1
    return x, y

x = paddle.to_tensor(1)
y = paddle.to_tensor(2)
out = func(x, y)
```

上述代码将报如下错误：`InvalidArgumentError: The type of data we are trying to retrieve does not match the type of data currently contained in the container.`

**修改建议**：对不同的类型可使用额外的变量，将功能进行拆分。

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


### 1.2 张量在不同分支 Shape 须保持一致

依赖控制流的 if...else 语句在动转静生成中间表示 Program 时，要求两个分支中同名张量的Shape必须保持一致，因为静态图下会对两个分支的输出进行动态 select input 操作，故须保证无论条件变量 x > y取何值，选取的张量 Shape 都是一致的。否则在后续组网或者训练时，出现因 Shape 不同而报错。

如下是一个错误的代码样例：

```python
# 错误例子：
import paddle
from paddle.jit import to_static

@to_static
def fun(x, y):
    z = paddle.randn([2, 3])

    if x > y:
        y = paddle.randn([2, 2])          # <--- y.shape 是[2, 2]
    else:
        y = paddle.randn([4, 5])          # <--- y.shape 是[4, 5]
    
    out = paddle.concat([y, z], axis=-1)  # <--- y 与 z 不能保证始终能 concat
    return out

x = paddle.to_tensor(1)
y = paddle.to_tensor(2)
out = fun(x, y)
```

上述代码将报如下错误：`InvalidArgumentError: The 0-th dimension of input[0] and input[1] is expected to be equal.But received input[0]'s shape = [4, 5], input[1]'s shape = [2, 3].`

修改建议：调整依赖控制流的 if...else 不同分支同名张量的代码逻辑，确保shape保持一致

### 2. for、while语句

#### 2.1 条件变量类型须保持不变

While 的条件变量在循环过程中的类型应保持不变，因为循环变量的类型将会决定其是保持 Python语法运行，或是转为飞桨的 whilie_loop API。保持条件变量类型不变才能确保模型正确地被动转静。

如下是一个错误的代码样例

```python
import paddle
from paddle.jit import to_static

@to_static
def func(x : paddle.Tensor):
    t = 2
    while t < 10:               # <--- 初始为bool类型，循环一次后为 Tensor类型
        t = paddle.shape(x)[0]  # <--- t 变为了 Tensor 类型
        x = paddle.concat([x, x])
    return x

x = paddle.randn([2, 3])
out = func(x)
```
如上述样例在执行循环体后，条件变量从 Python 的 bool 类型变为了 Tensor类型，动转静报错机制会捕捉并抛出异常：Dygraph2StaticException: python while pred change from bool to Tensor. 。

此处根据期望模型代码运行的效果，有如下两种「规范性」的写法：

* 若此处是一个循环次数固定的 while，则应避免t的类型变化，调整方式为：

```python
def func(x : paddle.Tensor):
    t = 2
    while t < 10:
        t = x.shape[0]   # <--- 借助x.shape 获取 int 类型值
        x = paddle.concat([x, x])
    return x
```

+ 若此处是一个循环次数不固定的while，则可以将t的类型提前转为Tensor，调整方式为：

```python
def func(x : paddle.Tensor):
    t = paddle.to_tensor(2)   # <--- 提前转为 Tensor 类型
    while t >= 0:
        t = paddle.shape(x)[0]
        x = paddle.concat([x, x])
    return x
```
对于for i in y的情况，可以调整为for i in paddle.to_tensor(y)即可。

#### 2.2 变量类型结构须保持一致

动态图模型中的for、while语句在动转静时，当其属于依赖控制流的情况时，其循环体内代码块逻辑只会被执行一次。若循环前后的同名变量的数据类型、数据结构不一致，将无法保证生成的中间表示 Program 涵盖动态图下所有逻辑，出现转写不等价的情况。

常见的错误有循环前为 None，循环后为 Tensor。这类情况建议使用默认值方法或者是 do-while 形式替换。如下是一个错误的代码样例：

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

上面的例子中，cache 在第一轮循环中为None，在后续的轮次中为Tensor，而目前无法同时表示None与Tensor 导致动转静后执行结果错误。

修改建议：可以将循环代码修改为如下的 do-while 形式，确保循环前后的数据类型一致性


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

#### 2.3 迭代变量Shape须保持不变
while 循环体中迭代变量的值可以变化，但是其shape须保持不变，否则可能导致隐式的错误（精度对不齐、结果错误之类）。若无法避免，可以通过动态 shape来解决。

如下是一个典型样例：
```python
import paddle
from paddle.jit import to_static

@to_static
def func(x, y):
    for i in range(paddle.to_tensor(3)):
        x = paddle.concat([x, y], axis=0)

    print(x.shape)    # <--- 动态图下返回 [8, 3], 静态图下返回[4, 3]
    # .....           # <--- 若此处存在依赖x.shape[0] 的代码逻辑，存在隐式错误
    return x

x = paddle.randn([2, 3])
y = paddle.randn([2, 3])
out = func(x, y)
print(out.shape)
```

```python

```