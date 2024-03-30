## [ 参数不一致 ]torch.min
输入一个 Tensor 对应 paddle.min，输入两个 Tensor 对应 paddle.minimum，因此有两组差异分析，分别如下：

--------------------------------------------------------------------------------------------------
### [torch.min](https://pytorch.org/docs/stable/generated/torch.min.html?highlight=min#torch.min)

```python
torch.min(input,
          dim=None,
          keepdim=False,
          *,
          out=None)
```

### [paddle.min](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/min_cn.html#min)

```python
paddle.min(x,
           axis=None,
           keepdim=False,
           name=None)
```

其中 PyTorch 与 Paddle 指定 `dim` 后返回值不一致，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 输入的 Tensor ，仅参数名不一致。                                      |
| dim           | axis         | 求最小值运算的维度， 仅参数名不一致。                                      |
| keepdim       | keepdim      | 是否在输出 Tensor 中保留减小的维度。  |
| out           | -            | 表示输出的 Tensor ， Paddle 无此参数，需要转写。               |
| 返回值           | 返回值            | 表示返回结果，当指定 dim 后，PyTorch 会返回比较结果和元素索引， Paddle 不会返回元素索引，需要转写。               |

### 转写示例
#### out：指定输出
```python
# 对指定维度上的 Tensor 元素求最大值运算

# PyTorch 写法
torch.min(a, out=y)
# torch 在输入 dim 时，返回 (values, indices)，返回参数类型不一致

# Paddle 写法
paddle.assign(paddle.min(a), y)
```
#### 指定 dim 后的返回值
```python
# PyTorch 写法
result = torch.min(a, dim=1)

# Paddle 写法
result = torch.min(a, dim=1), torch.argmin(a, dim=1)
```

--------------------------------------------------------------------------------------------------

### [torch.min](https://pytorch.org/docs/stable/generated/torch.min.html?highlight=min#torch.min)

```python
torch.min(input,
          other,
          *,
          out=None)
```

### [paddle.minimum](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/minimum_cn.html#minimum)

```python
paddle.minimum(x,
               y,
               name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 输入的 Tensor。                                      |
| other         | y            | 输入的 Tensor。                                      |
| out           | -            | 表示输出的 Tensor ， Paddle 无此参数，需要转写。               |


### 转写示例
#### out：指定输出
```python
# 逐元素对比输入的两个 Tensor

# PyTorch 写法
torch.min(a, b, out=y)
# 在输入 other 时，比较 input 和 other 返回较大值

# Paddle 写法
paddle.assign(paddle.minimum(a, b), y)
```
