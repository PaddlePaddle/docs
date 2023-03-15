## [torch 参数更多 ]torch.ones_like
### [torch.ones_like](https://pytorch.org/docs/stable/generated/torch.ones_like.html?highlight=ones_like#torch.ones_like)

```python
torch.ones_like(input,
                *,
                dtype=None,
                layout=None,
                device=None,
                requires_grad=False,
                memory_format=torch.preserve_format)
```

### [paddle.ones_like](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/ones_like_cn.html#ones-like)

```python
paddle.ones_like(x,
                 dtype=None,
                 name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 表示输入 Tensor。                                     |
| dtype         | dtype            | 表示数据类型。                                     |
| <font color='red'> layout </font> | -       | 表示布局方式， Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。  |
| <font color='red'> device </font>     | -       | 表示 Tensor 存放设备位置，Paddle 无此参数，需要进行转写。 |
| <font color='red'> requires_grad </font> | -       | 表示是否计算梯度， Paddle 无此参数，需要进行转写。 |
| <font color='red'> memory_format </font> | -  | 表示内存格式， Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。|


### 转写示例
#### requires_grad：是否需要求反向梯度，需要修改该 Tensor 的 stop_gradient 属性
```python
# Pytorch 写法
x = torch.ones_like((3. 5)), requires_grad=True)

# Paddle 写法
x = paddle.ones_like([3, 5])
x.stop_gradient = False
```

#### device: Tensor 的设备
```python
# Pytorch  写法
torch.ones_like(x, device=torch.device('cpu'))

# Paddle  写法
y = paddle.ones_like(x)
y.cpu()
```
