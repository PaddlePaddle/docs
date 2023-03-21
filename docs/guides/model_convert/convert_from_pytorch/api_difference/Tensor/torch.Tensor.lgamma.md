## [ torch 参数更多 ]torch.Tensor.lgamma

同 torch.lgamma

### [torch.Tensor.lgamma](https://pytorch.org/docs/stable/generated/torch.lgamma.html#torch.lgamma)

```python
torch.Tensor.lgamma(input, 
                    *, 
                    out=None)
```

### [paddle.Tensor.lgamma](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/lgamma_cn.html)

```python
paddle.Tensor.lgamma(x, 
                     name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射
| PyTorch | PaddlePaddle | 备注                                                     |
| ------- | ------------ | -------------------------------------------------------- |
| input   | x            | 输入的多维 Tensor，仅参数名不同。                       |
| out     | -            | 表示输出的 Tensor，PaddlePaddle 无此参数，需要进行转写。 |


### 转写示例

#### out: 指定输出
```python
# Pytorch 写法
torch.lgamma(torch.tensor([-0.4, -0.2, 0.1, 0.3]), out = y)

# Paddle 写法
paddle.lgamma(paddle.to_tensor([-0.4, -0.2, 0.1, 0.3]))
```
