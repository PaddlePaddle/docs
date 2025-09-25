## [ torch 参数更多 ]torch.bernoulli

### [torch.bernoulli](https://pytorch.org/docs/stable/generated/torch.bernoulli.html#torch.bernoulli)

```python
torch.bernoulli(input,
                p=None,
                *,
                generator=None,
                out=None)
```

### [paddle.bernoulli](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/bernoulli_cn.html)

```python
paddle.bernoulli(x,
                 p=None,
                 name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input      | x  | 伯努利参数 Tensor，仅参数名不一致。  |
| p          | p  | 可选，伯努利参数 p。 |
| generator  | -  | 用于采样的伪随机数生成器， Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
| out        | -  | 表示输出的 Tensor，Paddle 无此参数，需要转写。    |


### 转写示例
#### out：指定输出
```python
# PyTorch 写法
torch.bernoulli(x, out=y)

# Paddle 写法
paddle.assign(paddle.bernoulli(x), y)
```
