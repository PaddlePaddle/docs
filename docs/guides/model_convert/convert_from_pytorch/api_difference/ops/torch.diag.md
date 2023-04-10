## [torch 参数更多 ]torch.diag
### [torch.diag](https://pytorch.org/docs/stable/generated/torch.diag.html?highlight=diag#torch.diag)

```python
torch.diag(input,
           diagonal=0,
           *,
           out=None)
```

### [paddle.diag](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/diag_cn.html)

```python
paddle.diag(x,
            offset=0,
            padding_value=0,
            name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'> input </font> | <font color='red'> x </font> | 表示输入的 Tensor ，仅参数名不一致。  |
| <font color='red'> diagonal </font>      | <font color='red'> offset </font>      | 对角线偏移量。正值表示上对角线，0 表示主对角线，负值表示下对角线。                |
| <font color='red'> out </font> | -  | 表示输出的 Tensor ， Paddle 无此参数，需要进行转写。    |
| -             | <font color='red'>padding_value</font> | 表示填充指定对角线以外的区域， PyTorch 无此参数， Paddle 保持默认即可 。               |

### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.diag(x, out=y)

# Paddle 写法
paddle.assign(paddle.diag(x), y)
```
