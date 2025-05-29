## [ torch 参数更多 ]torch.utils.data.DataLoader

### [torch.utils.data.DataLoader](https://pytorch.org/docs/stable/data.html?highlight=dataloader#torch.utils.data.DataLoader)
```python
torch.utils.data.DataLoader(dataset,
                            batch_size=1,
                            shuffle=False,
                            sampler=None,
                            batch_sampler=None,
                            num_workers=0,
                            collate_fn=None,
                            pin_memory=False,
                            drop_last=False,
                            timeout=0,
                            worker_init_fn=None,
                            multiprocessing_context=None,
                            generator=None,
                            *,
                            prefetch_factor=2,
                            persistent_workers=False,
                            pin_memory_device='')
```

### [paddle.io.DataLoader](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/io/DataLoader_cn.html#dataloader)
```python
paddle.io.DataLoader(dataset,
                     feed_list=None,
                     places=None,
                     return_list=False,
                     batch_sampler=None,
                     batch_size=1,
                     shuffle=False,
                     drop_last=False,
                     collate_fn=None,
                     num_workers=0,
                     use_buffer_reader=True,
                     use_shared_memory=False,
                     timeout=0,
                     worker_init_fn=None)
```


### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| dataset       | dataset            | 表示数据集。  |
| batch_size       | batch_size            | 每 mini-batch 中样本个数。  |
| shuffle      | shuffle            | 生成 mini-batch 索引列表时是否对索引打乱顺序。  |
| sampler       | -            | 表示数据集采集器，Paddle 无此参数，暂无转写方式。  |
| batch_sampler      | batch_sampler            | mini-batch 索引列表。  |
| num_workers      | num_workers            | 用于加载数据的子进程个数。  |
| collate_fn      | collate_fn            | 用于指定如何将样本列表组合为 mini-batch 数据。  |
| pin_memory    | -            | 表示数据最开始是属于锁页内存，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
| drop_last    | drop_last            | 是否丢弃因数据集样本数不能被 batch_size 整除而产生的最后一个不完整的 mini-batch。 |
| timeout    | timeout           | 从子进程输出队列获取 mini-batch 数据的超时时间。 |
| worker_init_fn    | worker_init_fn           | 子进程初始化函数。 |
| multiprocessing_context | -  | 用于设置多进程的上下文，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。                   |
| generator     | -            | 用于采样的伪随机数生成器，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。   |
| prefetch_factor | -          | 表示每个 worker 预先加载的数据数量，Paddle 无此参数，暂无转写方式。  |
| persistent_workers | -       | 表示数据集使用一次后，数据加载器将会不会关闭工作进程，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。  |
| pin_memory_device  | -       | 数据加载器是否在返回 Tensor 之前将 Tensor 复制到设备固定存储器中，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。  |
| -             | feed_list    | 表示 feed 变量列表，PyTorch 无此参数，Paddle 保持默认即可                   |
| -             | places       | 数据需要放置到的 Place 列表，PyTorch 无此参数，Paddle 保持默认即可                   |
| -             | return_list  | 每个设备上的数据是否以 list 形式返回，PyTorch 无此参数，Paddle 保持默认即可                   |
| -             | use_buffer_reader | 表示是否使用缓存读取器，PyTorch 无此参数，Paddle 保持默认即可                   |
| -             | use_shared_memory | 表示是否使用共享内存来提升子进程将数据放入进程间队列的速度，PyTorch 无此参数，Paddle 保持默认即可   |
