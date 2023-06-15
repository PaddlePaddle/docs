## [ 参数不一致 ]torch.all

### [torch.all](https://pytorch.org/docs/stable/generated/torch.all.html?highlight=all#torch.all)

```python
torch.all(input)
```

### [paddle.all](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/all_cn.html#all)

```python
paddle.all(x,
           axis=None,
           keepdim=False,
           name=None)
```

其中 Paddle 和 PyTorch 的 `input` 参数所支持的数据类型不一致，具体如下：

### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'> input </font>         | <font color='red'> x </font>            | 输入的多维 Tensor ，PyTorch 支持其他类型的 Tensor ， Paddle 仅支持 bool 类型的 Tensor ，需要进行转写。           |
| -             | <font color='red'> axis </font>         | 计算逻辑与运算的维度，Pytorch 无，保持默认即可。               |
| -             | <font color='red'> keepdim </font>      | 是否在输出 Tensor 中保留减小的维度，Pytorch 无，保持默认即可。  |

### 转写示例
#### input：输入 Number 类型
```python
# Pytorch 写法
result = torch.all(x)

# Paddle 写法
result = paddle.all(x.astype(dtype='bool'))
```
