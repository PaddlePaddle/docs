## [torch 参数更多 ]torch.pow

### [torch.pow](https://pytorch.org/docs/stable/generated/torch.pow.html?highlight=pow#torch.pow)

```python
torch.pow(input,
          exponent,
          *,
          out=None)
```

### [paddle.pow](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/pow_cn.html)

```python
paddle.pow(x,
           y,
           name=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input  | x  | 表示输入的 Tensor ，仅参数名不一致。  |
| exponent  |  y  | 表示输入的 Tensor ，仅参数名不一致。  |
| out  | -  | 表示输出的 Tensor ， Paddle 无此参数，需要转写。    |


### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.pow([3, 5], 2, out=y)

# Paddle 写法
paddle.assign(paddle.pow([3, 5], 2), y)
```
