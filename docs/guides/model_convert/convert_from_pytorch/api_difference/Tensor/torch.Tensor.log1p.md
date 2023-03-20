## [ torch参数更多 ]torch.Tensor.log1p

同torch.log1p

### [torch.Tensor.log1p](https://pytorch.org/docs/stable/generated/torch.log1p.html)

```python
torch.Tensor.log1p(input, 
                   *, 
                   out=None)
```

### [paddle.Tensor.log1p](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/log1p_cn.html#log1p)

```python
paddle.Tensor.log1p(x, 
                    name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                     |
| ------- | ------------ | -------------------------------------------------------- |
| input   | x            | 输入的多维 Tensor ，仅参数名不同。                       |
| out     | -            | 表示输出的 Tensor，PaddlePaddle 无此参数，需要进行转写。 |


### 转写示例

#### out:指定输出

```python
# Pytorch 写法
torch.log1p(torch.tensor([-1.0090, -0.9923, 1.0249, -0.5372, 0.2492]), out = y)

# Paddle 写法
paddle.log1p(paddle.to_tensor([-1.0090, -0.9923,  1.0249, -0.5372, 0.2492]))
```