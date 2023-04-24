## [torch 参数更多 ]torch.less

### [torch.less](https://pytorch.org/docs/stable/generated/torch.less.html#torch.less)

```python
torch.less(input, other, *, out=None)
```

### [paddle.less_than](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/less_than_cn.html)

```python
paddle.less_than(x, y, name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 表示输入的 Tensor ，仅参数名不一致。                     |
| other         | y            | 表示输入的 Tensor ，仅参数名不一致。                     |
| out           | -            | 表示输出的 Tensor ， Paddle 无此参数，需要进行转写。      |


### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.less([3, 5], 1., out=y)

# Paddle 写法
paddle.assign(paddle.less_than([3, 5], 1.), y)
```
