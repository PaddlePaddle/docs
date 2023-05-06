## [torch 参数更多 ]torch.masked_select
### [torch.masked_select](https://pytorch.org/docs/1.13/generated/torch.masked_select.html?highlight=masked_select#torch.masked_select)

```python
torch.masked_select(input,
                   mask,
                   *,
                   out=None)
```

### [paddle.masked_select](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/masked_select_cn.html#masked-select)

```python
paddle.masked_select(x,
                    mask,
                    name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'> input </font> | <font color='red'> x </font> | 表示输入的 Tensor ，仅参数名不一致。  |
| mask         | mask            | 表示用于索引的二进制掩码的 Tensor 。                                      |
| <font color='red'> out </font> | -  | 表示输出的 Tensor ， Paddle 无此参数，需要进行转写。    |


### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.masked_select(x, mask, out=y)

# Paddle 写法
paddle.assign(paddle.masked_select(x, mask), y)
```
