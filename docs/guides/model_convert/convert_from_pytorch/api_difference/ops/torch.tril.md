## [仅 torch 参数更多]torch.tril

### [torch.tril](https://pytorch.org/docs/stable/generated/torch.tril.html?highlight=tril#torch.tril)

```python
torch.tril(input,diagonal=0,*,out=None)
```

### [paddle.tril](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/tril_cn.html)

```python
paddle.tril(input,diagonal=0,name=None)
```

其中 Paddle 相比 PyTorch 支持更多其他参数，具体如下：

### 参数映射
|PyTorch|PaddlePaddle|备注|
| ------- | ------- | ------- |
|input|input|input tensor即输入的矩阵|
|diagonal|diagonal|指定的对角线，默认值是0，表示主对角线。如果diagonal>0，表示主对角线之上的对角线；如果diagonal<0，表示主对角线之下的对角线。|
|out||output tensor即输出的矩阵，Paddle无此参数，需要进行转写|

### 转写示例

```python
# Pytorch 写法
torch.tril(torch.tensor([[-1.08,-0.86],[1.5,1.4]]), out=y)

# Paddle 写法
paddle.assign(paddle.tril(paddle.to_tensor([[-1.08,-0.86],[1.5,1.4]])), y)
```
