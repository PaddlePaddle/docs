## [ torch 参数更多 ]torch.Tensor.log10

同 torch.log10

### [torch.Tensor.log10](https://pytorch.org/docs/stable/generated/torch.log10.html)

```python
torch.Tensor.log10(input, 
                   *, 
                   out=None)
```

### [paddle.Tensor.log10](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/log10_cn.html#log10)

```python
paddle.Tensor.log10(x, 
                    name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                     |
| ------- | ------------ | -------------------------------------------------------- |
| input   | x            | 输入的多维 Tensor ，仅参数名不同。                       |
| out     | -            | 表示输出的 Tensor，PaddlePaddle 无此参数，需要进行转写。 |


### 转写示例

#### out: 指定输出

```python
# Pytorch 写法
torch.log10(torch.tensor([0.5224, 0.9354, 0.7257, 0.1301, 0.2251]), out = y)

# Paddle 写法
paddle.log10(paddle.to_tensor([0.5224, 0.9354, 0.7257, 0.1301, 0.2251]))
```
