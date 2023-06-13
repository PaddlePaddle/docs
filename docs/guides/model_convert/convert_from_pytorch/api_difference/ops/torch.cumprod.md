## [torch 参数更多 ]torch.cumprod

### [torch.cumprod](https://pytorch.org/docs/stable/generated/torch.cumprod.html?highlight=cumprod#torch.cumprod)

```python
torch.cumprod(input,
              dim,
              *,
              dtype=None,
              out=None)
```

### [paddle.cumprod](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/cumprod_cn.html#cumprod)

```python
paddle.cumprod(x,
               dim=None,
               dtype=None,
               name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------|
| input         | x            | 输入的 Tensor ，仅参数名不同。                          |
| dim           | dim          | 指明需要累乘的维度，参数名相同。                         |
| dtype         | dtype        | 输出 Tensor 的数据类型，如果指定了，那么在执行操作之前，输入的 Tensor 将被转换为 dtype 类型，这对防止数据类型溢出非常有用，默认为 None，参数名相同。        |
| out           | -            | 表示输出的 Tensor ，Paddle 无此参数，需要进行转写。      |


### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.cumprod(input, dim=1, dtype='float64', out=y)

# Paddle 写法
paddle.assign(paddle.cumprod(input, dim=1, dtype='float64'), y)
```
