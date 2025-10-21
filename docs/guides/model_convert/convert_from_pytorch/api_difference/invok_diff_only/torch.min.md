## [ 仅 API 调用方式不一致 ]torch.min
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
