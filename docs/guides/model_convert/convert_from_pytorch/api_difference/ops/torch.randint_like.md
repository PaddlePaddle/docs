## [torch 参数更多 ]torch.randint_like

### [torch.randint_like](https://pytorch.org/docs/stable/generated/torch.randint_like.html?highlight=randint_like#torch.randint_like)

```python
torch.randint_like(input,
                   low=0,
                   high,
                     *,
                   dtype=None,
                   layout=torch.strided,
                   device=None,
                   requires_grad=False,
                   memory_format=torch.preserve_format)
```

### [paddle.randint_like](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/randint_like_cn.html)

```python
paddle.randint_like(x,
                    low=0,
                    high=None,
                    dtype=None,
                    name=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                         |
| ------------- | ------------ | ------------------------------------------------------------ |
| input         | x            | 表示输入的 Tensor ，仅参数名不一致。                         |
| low           | low          | 表示生成的随机值范围的下限(区间一般包含)。                   |
| high          | high         | 表示生成的随机值范围的上限(区间一般不包含)。                 |
| dtype         | dtype        | 表示数据类型。                                               |
| layout        | -            | 表示布局方式， Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
| device        | -            | 表示 Tensor 存放设备位置，Paddle 无此参数，需要转写。    |
| requires_grad | -            | 表示是否计算梯度， Paddle 无此参数，需要转写。           |
| memory_format | -            | 表示内存格式， Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |


### 转写示例

#### device: Tensor 的设备

```python
# Pytorch 写法
torch.randint_like(x, 10, device=torch.device('cpu'))

# Paddle 写法
y = paddle.randint_like(x, 10)
y.cpu()
```

#### requires_grad：是否求梯度

```python
# Pytorch 写法
x = torch.randint_like(x, 10, requires_grad=True)

# Paddle 写法
x = paddle.randint_like(x, 10)
x.stop_gradient = False
```
