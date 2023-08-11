## [torch 参数更多 ]torch.full_like
### [torch.full_like](https://pytorch.org/docs/stable/generated/torch.full_like.html?highlight=full_like#torch.full_like)

```python
torch.full_like(input,
                fill_value,
                *,
                dtype=None,
                layout=torch.strided,
                device=None,
                requires_grad=False,
                memory_format=torch.preserve_format)
```

### [paddle.full_like](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/full_like_cn.html#full-like)

```python
paddle.full_like(x,
                 fill_value,
                 dtype=None,
                 name=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'> input </font> | <font color='red'> x </font> | 表示输入的 Tensor ，仅参数名不一致。  |
| fill_value  |  fill_value  |  表示填充值。  |
| dtype | dtype  | 表示数据类型。|
| <font color='red'> layout </font> | -       | 表示布局方式， Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。  |
| <font color='red'> device </font>     | -       | 表示 Tensor 存放设备位置，Paddle 无此参数，需要转写。 |
| <font color='red'> requires_grad </font> | -       | 表示是否计算梯度， Paddle 无此参数，需要转写。 |
| <font color='red'> memory_format </font> | -  | 表示内存格式， Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。|


### 转写示例

#### device: Tensor 的设备
```python
# Pytorch 写法
torch.full_like(x, 1., device=torch.device('cpu'))

# Paddle 写法
y = paddle.full_like(x, 1.)
y.cpu()

#### requires_grad：是否需要求反向梯度，需要修改该 Tensor 的 stop_gradient 属性
```python
# Pytorch 写法
x = torch.full_like([3, 5], 1., requires_grad=True)

# Paddle 写法
x = paddle.full_like([3, 5], 1.)
x.stop_gradient = False
```
