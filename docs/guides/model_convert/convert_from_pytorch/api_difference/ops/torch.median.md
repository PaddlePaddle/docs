## [torch 参数更多]torch.median
### [torch.median](https://pytorch.org/docs/stable/generated/torch.median.html?highlight=median#torch.median)

```python
torch.median(input,
             dim=- 1,
             keepdim=False,
             *,
             out=None)
```

### [paddle.median](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/median_cn.html#median)

```python
paddle.median(x,
              axis=None,
              keepdim=False,
              name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'> input </font> | <font color='red'> x </font> | 表示输入的 Tensor ，仅参数名不一致。  |
| <font color='red'> dim </font> | <font color='red'> axis </font> | 表示进行运算的轴，仅参数名不一致。  |
| keepdim       | keepdim      | 是否在输出 Tensor 中保留减小的维度。  |
| <font color='red'> out </font> | -  | 表示输出的 Tensor ， Paddle 无此参数，需要进行转写。    |


### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.median([3, 5], dim=0, out=y)

# Paddle 写法
y = paddle.median([3, 5], axis=0)
```
