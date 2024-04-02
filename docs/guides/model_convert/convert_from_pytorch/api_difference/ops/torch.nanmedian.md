## [torch 参数更多]torch.nanmedian
### [torch.nanmedian](https://pytorch.org/docs/stable/generated/torch.nanmedian.html?highlight=nanmedian#torch.nanmedian)

```python
torch.nanmedian(input,
                dim=-1,
                keepdim=False,
                *,
                out=None)
```

### [paddle.nanmedian](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nanmedian_cn.html#nanmedian)

```python
paddle.nanmedian(x, axis=None, keepdim=False, mode='avg', name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'> input </font> | <font color='red'> x </font> | 表示输入的 Tensor ，仅参数名不一致。  |
| <font color='red'> dim </font> | <font color='red'> axis </font> | 表示进行运算的轴，仅参数名不一致。  |
| keepdim       | keepdim         | 表示是否在输出 Tensor 中保留减小的维度。               |
| -             | mode         | 当 x 在所需要计算的轴上有偶数个非 NaN 元素时，选择使用平均值或最小值确定非 NaN 中位数的值， PyTorch 无此参数，Paddle 需设置为 'min'。 |
| <font color='red'> out </font> | -  | 表示输出的 Tensor ， Paddle 无此参数，需要转写。    |


### 转写示例

#### out：指定输出
```python
# PyTorch 写法
torch.nanmedian(a, -1, out=y)

# Paddle 写法
paddle.assign(paddle.nanmedian(a, -1), y)
```
