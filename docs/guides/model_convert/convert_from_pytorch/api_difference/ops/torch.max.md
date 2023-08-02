## [ 参数不一致 ]torch.max
输入一个 Tensor 对应 paddle.max，输入两个 Tensor 对应 paddle.maximum，因此有两组差异分析，分别如下：

--------------------------------------------------------------------------------------------------
### [torch.max](https://pytorch.org/docs/stable/generated/torch.max.html?highlight=max#torch.max)

```python
torch.max(input,
          dim=None,
          keepdim=False,
          *,
          out=None)
```

### [paddle.max](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/max_cn.html#max)

```python
paddle.max(x,
           axis=None,
           keepdim=False,
           name=None)
```

其中 Pytorch 与 Paddle 指定 `dim` 后返回值不一致，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 输入的 Tensor ，仅参数名不一致。                                      |
| dim           | axis         | 求最大值运算的维度， 仅参数名不一致。                                      |
| keepdim       | keepdim      | 是否在输出 Tensor 中保留减小的维度。  |
| out           | -            | 表示输出的 Tensor ， Paddle 无此参数，需要转写。               |
| 返回值           | 返回值            | 表示返回结果，当指定 dim 后，PyTorch 会返回比较结果和元素索引， Paddle 不会返回元素索引，需要转写。               |


### 转写示例
#### out：指定输出
```python
# 对指定维度上的 Tensor 元素求最大值运算

# Pytorch 写法
torch.max(a, out=y)
# torch 在输入 dim 时，返回 (values, indices)，返回参数类型不一致

# Paddle 写法
paddle.assign(paddle.max(a), y)
```
#### 指定 dim 后的返回值
```python
# Pytorch 写法
result = torch.max(a, dim=1)

# Paddle 写法
result = torch.max(a, dim=1), torch.argmax(a, dim=1)
```

--------------------------------------------------------------------------------------------------

### [torch.max](https://pytorch.org/docs/stable/generated/torch.max.html?highlight=max#torch.max)

```python
torch.max(input,
          other,
          *,
          out=None)
```

### [paddle.maximum](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/minimum_cn.html#minimum)

```python
paddle.maximum(x,
               y,
               name=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 输入的 Tensor ，仅参数名不一致。                                      |
| other         | y            | 输入的 Tensor ， 仅参数名不一致。                                      |
| out           | -            | 表示输出的 Tensor ， Paddle 无此参数，需要转写。               |


### 转写示例
#### out：指定输出
```python
# 逐元素对比输入的两个 Tensor

# Pytorch 写法
torch.max(a, b, out=y)
# 在输入 other 时，比较 input 和 other 返回较大值

# Paddle 写法
paddle.assign(paddle.maximum(a, b), y)
```
