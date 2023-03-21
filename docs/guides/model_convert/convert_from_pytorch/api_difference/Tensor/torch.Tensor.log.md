## [ torch 参数更多 ]torch.Tensor.log

同 torch.log

### [torch.Tensor.log](https://pytorch.org/docs/stable/generated/torch.Tensor.log.html)

```python
torch.Tensor.log(input, 
                 *, 
                 out=None)
```

### [paddle.Tensor.log](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/log_cn.html#log)

```python
paddle.Tensor.log(x, 
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
torch.log(torch.tensor([4.7767, 4.3234, 1.2156, 0.2411, 4.5739]), out = y)

# Paddle 写法
paddle.log(paddle.to_tensor([4.7767, 4.3234, 1.2156, 0.2411, 4.5739]))
```
