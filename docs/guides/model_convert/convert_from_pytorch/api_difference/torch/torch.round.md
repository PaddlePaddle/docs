## [torch 参数更多 ]torch.round
### [torch.round](https://pytorch.org/docs/stable/generated/torch.round.html?highlight=round#torch.round)

```python
torch.round(input,
            *,
            decimals=0,
            out=None)
```

### [paddle.round](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/round_cn.html#round)

```python
paddle.round(x,
             name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 输入的 Tensor ，仅参数名不一致。                                      |
| decimals      | decimals     | 要舍入到的小数位数。               |
| out           | -            | 表示输出的 Tensor，Paddle 无此参数，需要转写。               |


### 转写示例
#### out：指定输出
```python
# PyTorch 写法
x = torch.tensor([3, 5])
torch.round(x, out=y)

# Paddle 写法
x = paddle.to_tensor([3, 5])
paddle.assign(paddle.round(x), y)
```
