## torch.masked_select
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
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 输入的 Tensor。                                      |
| out           | -            | 表示输出的 Tensor，PaddlePaddle 无此参数。               |


### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.masked_select(x, mask, out=y)

# Paddle 写法
y = paddle.masked_select(x, mask)
```
