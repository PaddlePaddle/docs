## [torch 参数更多]torch.zeros_like

###  [torch.zeros_like](https://pytorch.org/docs/stable/generated/torch.zeros_like.html?highlight=zeros_like#torch.zeros_like)

```python
torch.zeros_like(input,
                 *,
                 dtype=None,
                 layout=None,
                 device=None,
                 requires_grad=False,
                 memory_format=torch.preserve_format)
```

###  [paddle.zeros_like](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/zeros_like_cn.html)

```python
paddle.zeros_like(x,
                  dtype=None,
                  name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                         |
| :------------ | :----------- | :----------------------------------------------------------- |
| input         | x            | 表示输入 Tensor ，仅名称不同。                               |
| dtype         | dtype        | 表示输出 Tensor 类型。                                       |
| layout        | -            | 表示布局方式，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
| device        | -            | 表示 Tensor 存放设备位置，Paddle 无此参数，需要转写。    |
| requires_grad | -            | 表示是否计算梯度，Paddle 无此参数，需要转写。            |
| memory_format | -            | 表示内存格式，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |

### 转写示例

#### device: Tensor 的设备

```python
# PyTorch 写法
y = torch.zeros_like(x, device=torch.device('cpu'))

# Paddle 写法
y = paddle.zeros_like(x)
y.cpu()
```

#### requires_grad：是否求梯度

```python
# PyTorch 写法
y = torch.zeros_like(x, requires_grad=True)

# Paddle 写法
y = paddle.zeros_like(x)
y.stop_gradient = False
```
