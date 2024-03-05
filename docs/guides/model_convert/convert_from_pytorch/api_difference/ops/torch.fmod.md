## [torch 参数更多 ]torch.fmod
### [torch.fmod](https://pytorch.org/docs/stable/generated/torch.fmod.html?highlight=fmod#torch.fmod)

```python
torch.fmod(input,
           other,
           *,
           out=None)
```

### [paddle.mod](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/mod_cn.html#mod)

```python
paddle.mod(x,
           y,
           name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'> input </font> | <font color='red'> x </font> | 表示输入的被除数 ，仅参数名不一致。  |
| <font color='red'> other </font> | <font color='red'> y </font> | 表示输入的除数， PyTorch 可以为 Tensor 或 scalar，Paddle 只能为 Tensor 。  |
| <font color='red'> out </font> | -  | 表示输出的 Tensor ， Paddle 无此参数，需要转写。    |


### 转写示例
#### out：指定输出
```python
# PyTorch 写法
torch.fmod([3, 5], [1, 2], out=y)

# Paddle 写法
paddle.assign(paddle.mod([3, 5], [1, 2]), y)
```
