## [torch 参数更多 ]torch.max
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

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 输入的 Tensor ，仅参数名不一致。                                      |
| dim           | axis         | 求最大值运算的维度， 仅参数名不一致。                                      |
| keepdim       | keepdim      | 是否在输出 Tensor 中保留减小的维度。  |
| out           | -            | 表示输出的 Tensor ， Paddle 无此参数，需要进行转写。               |


### 转写示例
#### out：指定输出
```python
# 对指定维度上的 Tensor 元素求最大值运算

# Pytorch 写法
torch.max(a, 1, out=y)
# 在输入 dim 时，返回 (values, indices)

# Paddle 写法
y = paddle.max(a, 1)
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

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 输入的 Tensor ，仅参数名不一致。                                      |
| other         | y            | 输入的 Tensor ， 仅参数名不一致。                                      |
| out           | -            | 表示输出的 Tensor ， Paddle 无此参数，需要进行转写。               |


### 转写示例
#### out：指定输出
```python
# 逐元素对比输入的两个 Tensor

# Pytorch 写法
torch.max(a, b, out=y)
# 在输入 other 时，比较 input 和 other 返回较大值

# Paddle 写法
y = paddle.maximum(a, b)
```
