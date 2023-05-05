## [仅 torch 参数更多]torch.tril_indices

### [torch.tril_indices](https://pytorch.org/docs/stable/generated/torch.tril_indices.html?highlight=tril_indices#torch.tril_indices)

```python
torch.tril_indices(row,col,offset=0,*,dtype=torch.long,device='cpu',layout=torch.strided)
```

### [paddle.tril_indices](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/tril_indices_cn.html)

```python
paddle.tril_indices(row,col,offset=0,dtype='int64')
```

其中 Paddle 相比 PyTorch 支持更多其他参数，具体如下：

### 参数映射
|PyTorch|PaddlePaddle|备注|
| ------- | ------- | ------- |
|row|row|输入矩阵行数|
|col|col|输入矩阵列数|
|offset|offset| 确定从指定二维平面中获取对角线的位置。如果offset=0，取主对角线；如果offset>0，取主对角线右上的对角线；如果offset<0，取主对角线左下的对角线|
|dtype|dtype|输出张量的数据类型|
|device||输出张量的设备，默认使用当前设备|
|layout||输出张量的layout，只支持torch.strided|

### 转写示例

```python
# Pytorch 写法
>>> a = torch.tril_indices(3, 3)
>>> a
tensor([[0, 1, 1, 2, 2, 2],
        [0, 0, 1, 0, 1, 2]])

# Paddle 写法
 
import paddle

# example 1, default offset value
data1 = paddle.tril_indices(4,4,0)
print(data1)
# [[0, 1, 1, 2, 2, 2, 3, 3, 3, 3],
#  [0, 0, 1, 0, 1, 2, 0, 1, 2, 3]]

```
