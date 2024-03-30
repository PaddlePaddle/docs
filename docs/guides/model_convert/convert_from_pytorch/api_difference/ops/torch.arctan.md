## [torch 参数更多 ]torch.arctan

### [torch.arctan](https://pytorch.org/docs/stable/generated/torch.arctan.html#torch.arctan)

```python
torch.arctan(input,
             *,
             out=None)
```

### [paddle.atan](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/atan_cn.html#atan)

```python
paddle.atan(x,
            name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 输入的 Tensor ，仅参数名不一致。                                      |
| out           | -            | 表示输出的 Tensor，Paddle 无此参数，需要转写。               |


### 转写示例
#### out：指定输出
```python
# PyTorch 写法
torch.arctan(torch.tensor([0.2341, 0.2539]), out=y)

# Paddle 写法
paddle.assign(paddle.atan(paddle.to_tensor([0.2341, 0.2539])), y)
```
