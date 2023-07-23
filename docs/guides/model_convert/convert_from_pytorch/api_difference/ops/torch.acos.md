## [torch 参数更多 ]torch.acos

### [torch.acos](https://pytorch.org/docs/stable/generated/torch.acos.html?highlight=acos#torch.acos)

```python
torch.acos(input,
           *,
           out=None)
```

### [paddle.acos](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/acos_cn.html#acos)

```python
paddle.acos(x,
            name=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                     |
| ------- | :----------- | -------------------------------------------------------- |
| input   | x            | 表示输入的 Tensor ，仅参数名不一致。                     |
| out     | -            | 表示输出的 Tensor，PaddlePaddle 无此参数，需要进行转写。 |

### 转写示例

#### out：指定输出

```python
# Pytorch 写法
torch.acos(torch.tensor([0.3348, -0.5889]), out=y)

# Paddle 写法
paddle.assign(paddle.acos(paddle.to_tensor([0.3348, -0.5889])), y)

```
