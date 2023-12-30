## [torch 参数更多 ]torch.lt

### [torch.lt](https://pytorch.org/docs/stable/generated/torch.lt.html#torch.lt)

```python
torch.lt(input, other, *, out=None)
```

### [paddle.less_than](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/less_than_cn.html)

```python
paddle.less_than(x, y, name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 表示输入的 Tensor ，仅参数名不一致。                     |
| other         | y            | 表示输入的 Tensor ，仅参数名不一致。                     |
| out           | -            | 表示输出的 Tensor ， Paddle 无此参数，需要转写。      |


### 转写示例
#### out：指定输出
```python
# PyTorch 写法
torch.lt(x, y, out=output)

# Paddle 写法
paddle.assign(paddle.less_than(x,y), output)
```
