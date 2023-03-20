## [ torch参数更多 ]torch.Tensor.log2

同torch.log2

### [torch.Tensor.log2](https://pytorch.org/docs/stable/generated/torch.log2.html)

```python
torch.Tensor.log2(input, 
                  *, 
                  out=None)
```

### [paddle.Tensor.log2](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/log2_cn.html#log2)

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
torch.log2(torch.tensor([0.8419, 0.8003, 0.9971, 0.5287, 0.0490]), out = y)

# Paddle 写法
paddle.log2(paddle.to_tensor([0.8419, 0.8003, 0.9971, 0.5287, 0.0490]))
```