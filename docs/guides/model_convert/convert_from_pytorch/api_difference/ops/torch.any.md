## [ 参数不一致 ]torch.any

### [torch.any](https://pytorch.org/docs/stable/generated/torch.any.html?highlight=any#torch.any)

```python
torch.any(input)
```

### [paddle.any](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/any_cn.html#any)

```python
paddle.any(x,
           axis=None,
           keepdim=False,
           name=None)
```

其中 Paddle 与 Pytorch 的 `input` 参数所支持的类型不一致，具体如下：

### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input        | x           | 输入的多维 Tensor ，PyTorch 支持布尔和数值类型的输入，Paddle 仅支持布尔类型，需要进行转写。                   |
| -    | <font color='red'> axis </font>     | 计算逻辑与运算的维度，Pytorch 无此参数，保持默认即可。        |
| -    | <font color='red'> keepdim </font>| 是否在输出 Tensor 中保留减小的维度，Pytorch 无此参数，保持默认即可。  |

### 转写示例
```python
# PyTorch 写法
y = torch.any(x)

# Paddle 写法
y = paddle.any(x.astype('bool'))
```
