## [torch 参数更多 ]torch.minimum

### [torch.minimum](https://pytorch.org/docs/stable/generated/torch.minimum.html#torch.minimum)

```python
torch.minimum(input, other, *, out=None)
```

### [paddle.minimum](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/minimum_cn.html)

```python
paddle.minimum(x, y, name=None)
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
torch.minimum(x, y, out=output)

# Paddle 写法
paddle.assign(paddle.minimum(x,y), output)
```
