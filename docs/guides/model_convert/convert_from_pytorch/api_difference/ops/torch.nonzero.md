## [torch 参数更多 ]torch.nonzero
### [torch.nonzero](https://pytorch.org/docs/1.13/generated/torch.nonzero.html?highlight=nonzero#torch.nonzero)

```python
torch.nonzero(input,
              *,
              out=None,
              as_tuple=False)
```

### [paddle.nonzero](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nonzero_cn.html#nonzero)

```python
paddle.nonzero(x,
               as_tuple=False)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'> input </font> | <font color='red'> x </font> | 表示输入的 Tensor ，仅参数名不一致。  |
| <font color='red'> out </font> | -  | 表示输出的 Tensor ， Paddle 无此参数，需要进行转写。    |
| as_tuple | as_tuple  |  表示是否以元组格式返回。   |


### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.nonzero(x, out=y)

# Paddle 写法
y = paddle.nonzero(x)
```
