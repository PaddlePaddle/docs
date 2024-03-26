## [torch 参数更多 ]torch.fmin

### [torch.fmin](https://pytorch.org/docs/stable/generated/torch.fmin.html#torch.fmin)

```python
torch.fmin(input, other, *, out=None)
```

### [paddle.fmin](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/fmin_cn.html)

```python
paddle.fmin(x, y, name=None)
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
torch.fmin(x, y, out=output)

# Paddle 写法
paddle.assign(paddle.fmin(x,y), output)
```
