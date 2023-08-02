## [torch 参数更多 ]torch.maximum

### [torch.maximum](https://pytorch.org/docs/stable/generated/torch.maximum.html#torch.maximum)

```python
torch.maximum(input, other, *, out=None)
```

### [paddle.maximum](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/maximum_cn.html)

```python
paddle.maximum(x, y, name=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 表示输入的 Tensor ，仅参数名不一致。                     |
| other         | y            | 表示输入的 Tensor ，仅参数名不一致。                     |
| out           | -            | 表示输出的 Tensor ， Paddle 无此参数，需要转写。      |


### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.maximum(x, y, out=output)

# Paddle 写法
paddle.assign(paddle.maximum(x,y), output)
```
