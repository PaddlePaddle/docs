## [仅 torch 参数更多]torch.tensordot

### [torch.tensordot](https://pytorch.org/docs/stable/generated/torch.tensordot.html?highlight=tensordot#torch.tensordot)

```python
torch.tensordot(a,b,dims=2,out=None)
```

### [paddle.tensordot](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/tensordot_cn.html)

```python
paddle.tensordot(x,y,axes=2,name=None)
```

其中 Paddle 相比 PyTorch 支持更多其他参数，具体如下：

### 参数映射
|PyTorch|PaddlePaddle|备注|
| ------- | ------- | ------- |
|a|x|缩并运算的左张量|
|b|y|缩并运算的右张量|
|dims|axes|缩并运算的维度（轴），默认值为2|
|out||缩并运算的结果，Paddle无此参数，需要进行转写|

### 转写示例

```python
# Pytorch 写法
>>> a = torch.tensor([[-1.08,-0.86],[1.5,1.4]])
>>> b = torch.tensor([[-3,-4],[5,4]])
>>> torch.tensordot(a, b, dims=([1,0],[0,1]), out=z)

# Paddle 写法
x = paddle.to_tensor([[-1.08,-0.86],[1.5,1.4]])
y = paddle.to_tensor([[-3,-4],[5,4]])
paddle.assign(paddle.tensordot(x, y, axes=([1,0],[0,1])),z)
```
