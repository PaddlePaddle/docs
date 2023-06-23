## [ 组合替代实现 ]torch.cdist

### [torch.cdist](https://pytorch.org/docs/stable/generated/torch.cdist.html#torch.cdist)

```python
torch.cdist(x1, x2, p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary')
```
Paddle 无此 API，需要组合实现。

### 转写示例
#### 当 x1 为 2D tensor 时
```python
# Pytorch 写法
y = torch.cdist(x1, x2, p)

# Paddle 写法
dist_list = []
for i in range(x1.shape[0]):
    for j in range(x2.shape[0]):
        dist_list.append(paddle.dist(x1[i, :], x2[j, :], p=p).item())
y = paddle.to_tensor(dist_list).reshape([x1.shape[0], x2.shape[0]])
```

#### 当 x1 为 3D tensor 时
```python
# Pytorch 写法
y = torch.cdist(x1, x2, p)

# Paddle 写法
dist_list = []
for b in range(x1.shape[0]):
    for i in range(x1.shape[1]):
        for j in range(x2.shape[1]):
            dist_list.append(paddle.dist(x1[b, i, :], x2[b, j, :], p=p).item())
y = paddle.to_tensor(dist_list).reshape([x1.shape[0], x1.shape[1],x2.shape[1]])
```
