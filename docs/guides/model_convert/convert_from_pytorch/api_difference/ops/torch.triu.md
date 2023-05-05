## [仅 torch 参数更多]torch.triu

### [torch.triu](https://pytorch.org/docs/stable/generated/torch.triu.html?highlight=triu#torch.triu)

```python
torch.triu(input,diagonal=0,*,out=None)
```

### [paddle.triu](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/triu_cn.html)

```python
paddle.triu(input,diagonal=0,name=None)
```

其中 Paddle 相比 PyTorch 支持更多其他参数，具体如下：

### 参数映射
|PyTorch|PaddlePaddle|备注|
| ------- | ------- | ------- |
|input|input|输入矩阵|
|diagonal|diagonal|指定的对角线，默认值为0，表示主对角线。如果diagonal>0，表示主对角线之上的对角线；如果diagonal<0，表示主对角线之下的对角线|
|out||输出矩阵，Paddle没有此字段，需要进行转写|

### 转写示例

```python
# Pytorch 写法
torch.triu(torch.tensor([[-1.08,-0.86],[1.5,1.4]]),diagonal=1,out=y)


# Paddle 写法
x = paddle.to_tensor([[-1.08,-0.86],[1.5,1.4]])
out = paddle.triu(x,diagonal=1)
print(out)
```
