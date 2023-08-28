## [ 参数不一致 ]torch.less_equal

### [torch.less_equal](https://pytorch.org/docs/stable/generated/torch.less_equal.html#torch.less_equal)

```python
torch.less_equal(input, other, *, out=None)
```

### [paddle.less_equal](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/less_equal_cn.html)

```python
paddle.less_equal(x, y, name=None)
```

其中 Paddle 和 PyTorch 的 `other` 参数所支持的数据类型不一致，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 表示输入的 Tensor ，仅参数名不一致。                     |
| other         | y            | 表示输入的 Tensor ，PyTorch 支持 Number 和 Tensor 类型， Paddle 仅支持 Tensor 类型。当输入为 Number 类型时，需要转写。      |
| out           | -            | 表示输出的 Tensor ， Paddle 无此参数，需要转写。      |


### 转写示例
#### other：输入为 Number
```python
# Pytorch 写法
torch.less_equal(x, 2)

# Paddle 写法
paddle.less_equal(x, paddle.to_tensor(2))
```

#### out：指定输出
```python
# Pytorch 写法
torch.less_equal(x, y, out=output)

# Paddle 写法
paddle.assign(paddle.less_equal(x,y), output)
```
