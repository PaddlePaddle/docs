# 自动并行训练

超大模型已成为 AI 最重要的核心竞争力之一，近年来，国内外科技公司纷纷加大投入，模型规模迅速扩大，新的模型结构和算法创新不断涌现，分布式并行策略也日趋多样化。然而，当前的分布式训练框架使用门槛普遍较高。Megatron、DeepSpeed 等传统的分布式框架基于动态图接口，要求用户手动在模型组网中实现并行切分和通信逻辑，这对用户的底层硬件和通信操作理解提出了极高的要求，也大大增加了 AI 基建工程的复杂性。业界一些框架基于静态图实现自动并行，虽然静态图架构具备全局优化能力和方便推理部署的优势，但开发过程不够灵活、调试也不便，且在复杂网络结构下实现组网的编程难度较大，易用性存在明显短板。这些问题使得大模型训练成为少数高水平玩家的游戏，大量研究工作者因并行工程的高难度而无法推进算法开发，极大地制约了 AI 领域的创新和发展。

随着大模型技术的飞速发展，深度学习框架的架构设计也在经历着根本性变革。新一代的分布式训练框架需要兼顾研发效率和训练性能，只有提供灵活便捷的开发调试能力，才能支持模型的快速迭代。飞桨框架基于多年来对用户体验的持续积累和打磨，3.0 版本创新性地推出了动静统一的自动并行架构。自动并行架构提供了对分布式计算任务的统一抽象，让用户通过简单的张量切分标记即可实现不同的并行模式，极大地降低了分布式策略的开发门槛。通过融合动态图的灵活性与静态图的高效性，飞桨自动并行在国产框架中率先实现了“编码时动态调试，运行时静态优化”的分布式新编程范式。以 Llama 模型为例，使用飞桨自动并行框架实现分布式训练，并行工程相关代码可减少 50%，调试时间也大幅缩短，并可通过简单添加一行代码，实现从动态到静态的转换，自动应用框架多种静态优化技术，获得匹敌甚至超越经过极致手工优化的动态图表现。

## 一、分布式计算对象
一个计算任务通常需要包含两类实体：资源和程序，其中程序又由数据和计算指令构成。对于传统的串行深度学习任务，资源往往是单张计算卡，数据对应张量（Tensor），计算指令则对应算子（Operator，OP）。在分布式深度学习任务中，资源往往是由多张计算卡组成的同构或异构集群，张量需要在多张卡上进行切分，算子也需要具备多卡并行计算的能力，对应地需要分布式的张量和分布式的算子。飞桨自动并行架构通过对集群、张量和算子的统一分布式建模和抽象，使得用户能够方便地表示、构建和使用这些分布式计算实体，低成本地开发分布式深度学习任务。

### 1.1 分布式集群
我们使用 ProcessMesh 来对分布式计算集群进行抽象和表示。ProcessMesh 可以理解为一个高维矩阵，用来表示进程的笛卡尔拓扑（Cartesian Topology）关系，其中每个元素代表一个进程，它能任意 reshape 或 slice。在分布式训练任务中，往往一个进程对应一张计算卡。ProcessMesh 中包含 mesh 和 dim_names 两个属性，其中：

* mesh(list|numpy.array): 由网格中所有进程 id（process_ids）组成的列表或数组，表示一组设备的逻辑笛卡尔拓扑结构，笛卡尔网格的每个维度以名称进行引用。
* dim_names(list[str])：各个维度的名称，同一 ProcessMesh 内的网格维度名称必须唯一。在实际应用时，我们习惯以网格维度所对应的切分策略对该维度进行命名，例如做数据并行（Data Parallel, DP）的网格维度命名为'dp'，张量并行（Tensor Parallel, TP）的网格维度命名为'tp'等。
例如使用 2 卡做数据并行，我们可以用一个形状为(2)的一维矩阵来描述计算资源：

```python
mesh = paddle.distributed.ProcessMesh([0, 1], dim_names=['dp'])
```
使用 4 卡，可以表示成一个形状为(2, 2)的二维矩阵，其中不同的维度可以实现不同的并行策略：

```python
mesh = paddle.distributed.ProcessMesh([[0, 1], [2, 3]], dim_names=['dp', 'tp'])
```
类似地，4 机 32 卡的集群，可以使用一个形状为(4, 8)的二维矩阵来表示。

> 注：关于 ProcessMesh 更详细的使用说明，可以参考[ProcessMesh 接口文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distributed/ProcessMesh_cn.html)。在不考虑集群动态扩缩容的场景下，一个特定的分布式计算任务运行时所使用的计算设备往往是固定的，使用[set_mesh](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distributed/set_mesh_cn.html)和[get_mesh](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distributed/get_mesh_cn.html)接口可以方便地设置和获取全局的 mesh 信息。

### 1.2 分布式张量
分布式张量是由多个计算设备上的局部串行张量共同组成的全局逻辑张量。分布式张量具有 process_mesh 和 placecements 两个属性，其中 placements 用来描述数据在对应集群设备上的切分状态。placements 是由[ Replicate](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distributed/Replicate_cn.html#cn-api-paddle-distributed-replicate)、[Shard](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distributed/Shard_cn.html#cn-api-paddle-distributed-shard)和[Partial](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distributed/Partial_cn.html#cn-api-paddle-distributed-partial)三种分布式标记组成的列表，长度和 process_mesh 的维度个数一致，表示张量在对应的设备维度上，按照什么方式做切分。其中：

* Replicate()：指张量在所有计算设备上保持全复制状态。
* Shard(axis)：指张量沿 axis 维度进行切片，不同的分片放在不同的计算设备上。
* Partial(reduce_type)：指每个计算设备只拥有部分值，需要通过 allreduce_sum 或其它指定的规约操作才能恢复成全量数据。Partial 状态往往在网络运算过程中产生，用户很少需要显式标记 Partial 状态。

<figure align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/docs/develop/docs/guides/paddle_v3_features/images/auto_parallel/mesh.png" width="70%"/>
</figure>


假设有 6 个计算设备，我们将其表示成形状为(2, 3)的 ProcessMesh，其中编号 0、1、2 的设备为一组，编号 3、4、5 的设备为另一组，两个维度的名称分别为'x'和'y'：

```python
mesh = paddle.distributed.ProcessMesh([[0， 1， 2], [3, 4, 5]], dim_names=['x', 'y'])
```
在这 6 个设备上，我们想放置一个形状为(4, 3)的张量，其中计算设备的'x'维度切分张量的第 0 维，对应的分布式标记为 Shard(0)；'y'维度不做切分，对应的分布式标记为 Replicate()。使用[shard_tensor](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distributed/shard_tensor_cn.html#shard-tensor)接口，我们可以将一个串行的张量切分成分布式张量：

```python
dense_tensor = paddle.to_tensor([[1,2,3],
                                 [4,5,6],
                                 [7,8,9],
                                 [10,11,12]])

placements = [paddle.distributed.Shard(0), paddle.distributed.Replicate()]
dist_tensor = paddle.distributed.shard_tensor(dense_tensor, mesh, placements)

print(dist_tensor.process_mesh) # {shape: [2,3], process_ids: [0,1,2,3,4,5], dim_names: [x,y]}
print(dist_tensor.placements) # [Shard(dim=0), Replicate()]
```
<figure align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/docs/develop/docs/guides/paddle_v3_features/images/auto_parallel/shard.png" width="70%"/>
</figure>

对应的张量切分状态如上图所示，从数据行的维度（第 0 维）看，4 行数据被切分成了 2 块，每块 2 行，放置在设备的'x'维上，对应'x'维上的切分状态 Shard(0)。从数据列的维度看，数据没有做并行切分，'y'维上每个设备都拥有完整的 3 列数据，对应在'y'维上的全复制状态 Replicate()。

构建出分布式张量后，我们可以像写单卡程序一样，调用算子对分布式张量进行操作。当一个算子的输入中存在分布式张量时，其它的输入也都会被自动转换成分布式张量，对应的计算操作将在分布式的视角下进行，其输出结果也是分布式张量。框架底层会根据算子的计算逻辑自动进行必要的数据切分、并行计算和通信操作，以保证分布式计算结果的正确性。

有时候我们想在程序运算过程中人为改变张量的切分状态，以实现更好的并行效果，这时候可以使用[reshard](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distributed/reshard_cn.html#reshard)接口实现。例如，如果我们想让上述例子中分布式张量的第 1 个维度在计算设备的'y'维上做切分：

```python
placements = [paddle.distributed.Shard(0), paddle.distributed.Shard(1)]
dist_tensor = paddle.distributed.reshard(dist_tensor, mesh, placements)

print(dist_tensor.process_mesh) # {shape: [2,3], process_ids: [0,1,2,3,4,5], dim_names: [x,y]}
print(dist_tensor.placements) # [Shard(dim=0), Shard(dim=1)]
```
reshard 之后的张量切分状态如下，可以看到数据被切分得更细了：

<figure align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/docs/develop/docs/guides/paddle_v3_features/images/auto_parallel/reshard.png" width="70%"/>
</figure>

> 注：用户使用分布式张量的方式与普通的张量基本相同，在使用时不需要关心分布式算子的概念，因此此处不对分布式算子做展开介绍。如果你有开发新算子的需求，请查看[分布式算子开发](../../dev_guides/api_contributing_guides/auto_parallel_op_cn.md)章节内容。
>

## 二、基础并行策略
给定一个深度学习任务，其中的张量可以有多种不同的切分方式，不同的切分方式对应着不同的分布式策略。不同的分布式策略本质上是相同计算语义的不同分布式实现，大家熟悉的数据并行、张量并行、序列并行（Sequence Parallel, SP）、流水并行（Pipeline Parallel）、混合并行等，其实是基于专家经验从不同实现中人工挑选出来的典型分布式策略。

虽然分布式策略多种多样，但本质只有两类：SPMD（Single Program Multiple Data）和 Pipeline。前者关注一个张量在给定设备上如何切分，后者则聚焦于不同张量在不同设备上的切分。本节将以最常用的数据并行和模型并行为例子，向大家展示如何使用自动并行的分布式张量切分标记，方便而灵活地构建不同的 SPMD 策略。关于更复杂的 Pipeline 策略的构建方式，将在后文[流水并行](#pipeline)章节进行介绍。

### 2.1 数据并行实现
数据并行是最常用的分布式并行策略，这种策略将数据沿 batch 维切分成多份，每个计算资源都保存完整的模型参数并独立处理一份子数据集。使用自动并行，我们只需要将输入张量标记为沿着 batch 维做切分，则可自动进行数据并行训练，不需要进行其他额外的操作。

```python
# 运行方式： python train.py
import numpy as np
import paddle
from paddle.io import BatchSampler, DataLoader, Dataset

class RandomDataset(Dataset):
    def __init__(self, seq_len, hidden, num_samples=100):
        super().__init__()
        self.seq_len = seq_len
        self.hidden = hidden
        self.num_samples = num_samples

    def __getitem__(self, index):
        input = np.random.uniform(size=[self.seq_len, self.hidden]).astype("float32")
        label = np.random.uniform(size=[self.seq_len, self.hidden]).astype('float32')
        return (input, label)

    def __len__(self):
        return self.num_samples

class MlpModel(paddle.nn.Layer):
    def __init__(self):
        super(MlpModel, self).__init__()
        self.w0 = self.create_parameter(shape=[1024, 4096])
        self.w1 = self.create_parameter(shape=[4096, 1024])

    def forward(self, x):
        y = paddle.matmul(x, self.w0)
        z = paddle.matmul(y, self.w1)
        return z

with paddle.LazyGuard():
    model = MlpModel()
for p in model.parameters():
    p.initialize()

dataset = RandomDataset(128, 1024)
sampler = BatchSampler(
    dataset,
    batch_size=4,
)
dataloader = DataLoader(
    dataset,
    batch_sampler=sampler,
)
opt = paddle.optimizer.AdamW(learning_rate=0.001, parameters=model.parameters())
loss_fn = paddle.nn.MSELoss()

for step, (inputs, labels) in enumerate(dataloader):
    logits = model(inputs)
    loss = loss_fn(logits, labels)
    loss.backward()
    opt.step()
    opt.clear_grad()

print(f"max_memory_reserved = {paddle.device.cuda.max_memory_reserved() / 1e6 : .2f} MB") # 178.28 MB
```
上面的例子实现了一个由两个矩阵乘构成的单卡网络，并在运行之后打印出了程序的显存占用情况。假设我们想在 4 个计算设备上做数据运行，将输入数据沿第 0 维（batch 维）做切分。程序中每一条数据的大小为(128, 1024), batch_size 为 4，因此每一轮训练的总数据大小为（4, 128, 1024）。我们将数据切分到 4 个计算设备上，每个设备计算大小为（1, 128, 1024）的子数据。使用张量切分标记，只需要在程序中加入 3 行代码即可实现（第 4、第 7、第 30 行），不需要自己考虑切分和通信的逻辑：

```python
# 运行方式： python -m paddle.distributed.launch --device=0,1,2,3 train.py
import numpy as np
import paddle
import paddle.distributed as dist # 导入飞桨分布式训练模块
from paddle.io import BatchSampler, DataLoader, Dataset

mesh = dist.ProcessMesh([0, 1, 2, 3], dim_names=['dp']) # 定义计算资源

class RandomDataset(Dataset):
    def __init__(self, seq_len, hidden, num_samples=100):
        super().__init__()
        self.seq_len = seq_len
        self.hidden = hidden
        self.num_samples = num_samples

    def __getitem__(self, index):
        input = np.random.uniform(size=[self.seq_len, self.hidden]).astype("float32")
        label = np.random.uniform(size=[self.seq_len, self.hidden]).astype('float32')
        return (input, label)

    def __len__(self):
        return self.num_samples

class MlpModel(paddle.nn.Layer):
    def __init__(self):
        super(MlpModel, self).__init__()
        self.w0 = self.create_parameter(shape=[1024, 4096])
        self.w1 = self.create_parameter(shape=[4096, 1024])

    def forward(self, x):
        dist.shard_tensor(x, mesh, [dist.Shard(0)]) # 标记输入数据沿第 0 维切分
        y = paddle.matmul(x, self.w0)
        z = paddle.matmul(y, self.w1)
        return z

with paddle.LazyGuard():
    model = MlpModel()
for p in model.parameters():
    p.initialize()

dataset = RandomDataset(128, 1024)
sampler = BatchSampler(
    dataset,
    batch_size=4,
)
dataloader = DataLoader(
    dataset,
    batch_sampler=sampler,
)
opt = paddle.optimizer.AdamW(learning_rate=0.001, parameters=model.parameters())
loss_fn = paddle.nn.MSELoss()

for step, (inputs, labels) in enumerate(dataloader):
    logits = model(inputs)
    loss = loss_fn(logits, labels)
    loss.backward()
    opt.step()
    opt.clear_grad()

print(f"max_memory_allocated = {paddle.device.cuda.max_memory_allocated() / 1e6 : .2f} MB") # 176.16 MB
```
可以看到，做数据并行之后，程序的显存占用从 216.03MB 降到了 199.23MB。为了方便快速展示和体验，我们选取了一个 batch_size 和模型规模都较小的示例，在实际的大模型应用场景中，batch_size 对显存的影响往往要大得多，数据并行所带来的显存增益也会比示例大得多。此外，在实际应用场景中，由于模型优化器状态所消耗的显存较大，大家往往会在数据并行的设备维度对优化器状态也做并行切分，将优化器状态和数据一样分散在不同的设备上。通过[shard_optimizer](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distributed/shard_optimizer_cn.html#cn-api-paddle-distributed-shard-optimizer)接口，可以方便地切分模型的优化器状态，例如上面的程序中，我们可以在第 49 行定义优化器之后，调用 shard_optimizer 对优化器状态进行切分：

```python
opt = dist.shard_optimizer(opt)
```
除了在组网中对输入张量做切分，数据并行也可以直接通过[shard_dataloader](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distributed/shard_dataloader_cn.html)接口实现，在 dataloader 中根据需要直接完成对输入数据的切分：

```python
dataloader = paddle.distributed.shard_dataloader(dataloader, mesh, shard_dims="dp")
```


> 注：分布式训练程序需要通过[paddle.distributed.launch](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distributed/launch_cn.html#launch)多进程启动，运行时每张卡会有单独的输出日志，你可以在 log 目录下查看每张卡的日志，日志一般以 workerlog.xx 命名，其中 xx 为对应卡的逻辑编号。

### 2.2 张量并行实现
张量并行通过将模型参数切分到不同的计算设备，达到降低单个设备显存消耗的目的。张量并行的方式与具体的网络结构相关，是高度灵活和多样化的，例如常见的 Linear 结构，就有切分矩阵乘参数的第 0 维（行切）和第 1 维（列切）两种方式。使用自动并行，我们可以根据模型需要，灵活地对参数进行切分。例如在上面单卡网络的例子中，我们可以将第一个 matmul 的参数按列进行切分，第二个 matmul 的参数按行进行切分：

```python
# 运行方式： python -m paddle.distributed.launch --device=0,1,2,3 train.py
import numpy as np
import paddle
import paddle.distributed as dist # 导入飞桨分布式训练模块
from paddle.io import BatchSampler, DataLoader, Dataset

mesh = dist.ProcessMesh([0, 1, 2, 3], dim_names=['mp']) # 定义计算资源

class RandomDataset(Dataset):
    def __init__(self, seq_len, hidden, num_samples=100):
        super().__init__()
        self.seq_len = seq_len
        self.hidden = hidden
        self.num_samples = num_samples

    def __getitem__(self, index):
        input = np.random.uniform(size=[self.seq_len, self.hidden]).astype("float32")
        label = np.random.uniform(size=[self.seq_len, self.hidden]).astype('float32')
        return (input, label)

    def __len__(self):
        return self.num_samples

class MlpModel(paddle.nn.Layer):
    def __init__(self):
        super(MlpModel, self).__init__()
        self.w0 = self.create_parameter(shape=[1024, 4096])
        self.w1 = self.create_parameter(shape=[4096, 1024])

        self.w0 = dist.shard_tensor(self.w0, mesh, [dist.Shard(1)]) # 列切： w0 沿第 1 维切分
        self.w1 = dist.shard_tensor(self.w1, mesh, [dist.Shard(0)]) # 行切： w1 沿第 0 维切分

    def forward(self, x):
        y = paddle.matmul(x, self.w0)
        z = paddle.matmul(y, self.w1)
        return z

with paddle.LazyGuard():
    model = MlpModel()
for p in model.parameters():
    p.initialize()

dataset = RandomDataset(128, 1024)
sampler = BatchSampler(
    dataset,
    batch_size=4,
)
dataloader = DataLoader(
    dataset,
    batch_sampler=sampler,
)
opt = paddle.optimizer.AdamW(learning_rate=0.001, parameters=model.parameters())
loss_fn = paddle.nn.MSELoss()

for step, (inputs, labels) in enumerate(dataloader):
    logits = model(inputs)
    loss = loss_fn(logits, labels)
    loss.backward()
    opt.step()
    opt.clear_grad()

print(f"max_memory_reserved = {paddle.device.cuda.max_memory_reserved() / 1e6 : .2f} MB") # 119.56 MB
```
可以看到，将 Linear 参数做切分之后，单个设备的显存消耗从 216.03MB 降到了 127.95MB。更进一步地，如果我们使用 8 张计算设备，将数据并行和模型并行组合起来，可以构建出一个 DP2+MP4，带优化器状态切分的 2D 混合并行程序，单个设备的显存消耗将进一步降低到 82.86MB：

```python
# 运行方式： python -m paddle.distributed.launch --device=0,1,2,3,4,5,6,7 train.py
import numpy as np
import paddle
import paddle.distributed as dist # 导入飞桨分布式训练模块
from paddle.io import BatchSampler, DataLoader, Dataset

mesh = dist.ProcessMesh([[0, 1, 2, 3], [4, 5, 6, 7]], dim_names=['dp', 'mp']) # 定义计算资源, 8 张卡组织成(2, 4)的矩阵，外层 dp，内层 mp

class RandomDataset(Dataset):
    def __init__(self, seq_len, hidden, num_samples=100):
        super().__init__()
        self.seq_len = seq_len
        self.hidden = hidden
        self.num_samples = num_samples

    def __getitem__(self, index):
        input = np.random.uniform(size=[self.seq_len, self.hidden]).astype("float32")
        label = np.random.uniform(size=[self.seq_len, self.hidden]).astype('float32')
        return (input, label)

    def __len__(self):
        return self.num_samples

class MlpModel(paddle.nn.Layer):
    def __init__(self):
        super(MlpModel, self).__init__()
        self.w0 = self.create_parameter(shape=[1024, 4096])
        self.w1 = self.create_parameter(shape=[4096, 1024])

        self.w0 = dist.shard_tensor(self.w0, mesh, [dist.Replicate(), dist.Shard(1)]) # dp 维不切， mp 维列切： w0 沿第 1 维切分
        self.w1 = dist.shard_tensor(self.w1, mesh, [dist.Replicate(), dist.Shard(0)]) # dp 维不切，mp 维行切： w1 沿第 0 维切分

    def forward(self, x):
        x = dist.shard_tensor(x, mesh, [dist.Shard(0), dist.Replicate()]) # dp 维切分输入数据的第 0 维
        y = paddle.matmul(x, self.w0)
        z = paddle.matmul(y, self.w1)
        return z

with paddle.LazyGuard():
    model = MlpModel()
for p in model.parameters():
    p.initialize()

dataset = RandomDataset(128, 1024)
sampler = BatchSampler(
    dataset,
    batch_size=4,
)
dataloader = DataLoader(
    dataset,
    batch_sampler=sampler,
)

opt = paddle.optimizer.AdamW(learning_rate=0.001, parameters=model.parameters())
opt = dist.shard_optimizer(opt) # 切分优化器状态
loss_fn = paddle.nn.MSELoss()

for step, (inputs, labels) in enumerate(dataloader):
    logits = model(inputs)
    loss = loss_fn(logits, labels)
    loss.backward()
    opt.step()
    opt.clear_grad()

print(f"max_memory_reserved = {paddle.device.cuda.max_memory_reserved() / 1e6 : .2f} MB") # 84.93 MB
```

### 2.3 常见并行策略的接口封装
上面描述的分布式张量提供了一种高度灵活和方便的切分标记方式，为了简化用户使用分布式的各种并行策略，我们提供了更高层次的接口，支持在模型组网外配置数据并行、模型并行、流水并行等常见的并行策略，实现分布式代码和组网代码分离。例如，使用前面例子同样网络，DP2+MP4 的 2D 混合并行的程序如下：

```python
# 运行方式： python -m paddle.distributed.launch --device=0,1,2,3,4,5,6,7 train.py
import numpy as np
import paddle
import paddle.distributed as dist # 导入飞桨分布式训练模块
from paddle.io import BatchSampler, DataLoader, Dataset

mesh = dist.ProcessMesh([[0, 1, 2, 3], [4, 5, 6, 7]], dim_names=['dp', 'mp']) # 定义计算资源, 8 张卡组织成(2, 4)的矩阵，外层 dp，内层 mp
dist.set_mesh(mesh)

class RandomDataset(Dataset):
    def __init__(self, seq_len, hidden, num_samples=100):
        super().__init__()
        self.seq_len = seq_len
        self.hidden = hidden
        self.num_samples = num_samples

    def __getitem__(self, index):
        input = np.random.uniform(size=[self.seq_len, self.hidden]).astype("float32")
        label = np.random.uniform(size=[self.seq_len, self.hidden]).astype('float32')
        return (input, label)

    def __len__(self):
        return self.num_samples

class MlpModel(paddle.nn.Layer): # 模型组网中无需任何分布式代码，算法策略与分布式策略分离
    def __init__(self):
        super(MlpModel, self).__init__()
        self.w0 = paddle.nn.Linear(1024, 4096, bias_attr=False)
        self.w1 = paddle.nn.Linear(4096, 1024, bias_attr=False)

    def forward(self, x):
        y = self.w0(x)
        z = self.w1(y)
        return z

with paddle.LazyGuard():
    model = MlpModel()
opt = paddle.optimizer.AdamW(learning_rate=0.001, parameters=model.parameters())
parallel_config = {
    "dp_config": {'sharding_level': 1},
    "mp_config": {"parallelize_plan": {"w0": dist.ColWiseParallel(), "w1": dist.RowWiseParallel()}},
}
dist_model, dist_opt = dist.parallelize(model, opt, config=parallel_config)
for p in dist_model.parameters():
    p.initialize()

dataset = RandomDataset(128, 1024)
sampler = BatchSampler(
    dataset,
    batch_size=4,
)
dataloader = DataLoader(
    dataset,
    batch_sampler=sampler,
)
dataloader = dist.shard_dataloader(dataloader, meshes=[mesh], shard_dims="dp")
loss_fn = paddle.nn.MSELoss()

for step, (inputs, labels) in enumerate(dataloader()):
    logits = dist_model(inputs)
    loss = loss_fn(logits, labels)
    loss.backward()
    dist_opt.step()
    dist_opt.clear_grad()

print(f"max_memory_reserved = {paddle.device.cuda.max_memory_reserved() / 1e6 : .2f} MB") # 97.54 MB
```
可以看到，模型组网中无需任何分布式代码，算法策略可与分布式策略分离，简化开发且便于代码维护。通过 [paddle.distributed.parallelize ](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distributed/parallelize_cn.html#cn-api-paddle-distributed-parallelize)接口能够直观、容易地在模型组网外配置分布式策略并运行。

## 三、性能优化策略
在构建基础并行组网后，通过 [paddle.distributed.to_static](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distributed/to_static_cn.html#to-static) 接口，即可轻松实现从动态图模式到静态图模式的转换。在动态图转静态图（动转静）场景下，自动并行架构致力于保障用户的动静统一和极致性能体验。动静统一指的是相同的组网代码在动态图和静态图模式下能够具有相同的行为和结果，满足用户在动态图下调试和静态图下执行的体验一致性。极致性能指的是静态图模式相比动态图模式将提供更多内置的性能优化策略，基于静态图整图信息进行极致的性能优化，用户在静态图模式下可通过简单的配置开关开启相关优化，实现最佳的执行性能。

自动并行架构通过两个方面来保障动静统一的体验：

* 一是单卡组网的动静统一，自动并行架构直接对带切分标记的单卡组网进行动转静，避免分布式的复杂逻辑可能导致的动静不一致问题。
* 二是动态图和静态图架构内核的动静统一，在单卡组网动转静之后，动态图和静态图模式均由统一的内核进行切分推导和通信转换，这些关键的模块逻辑在动态图和静态图两种模式进行了统一的抽象，动静在各自的执行流程中执行相同的规则函数，实现相同的分布式逻辑。

<figure align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/docs/develop/docs/guides/paddle_v3_features/images/auto_parallel/to_static.png" width="70%"/>
</figure>


极致性能来自静态图下提前开发好的大量通用优化策略，包括张量重计算、算子融合、计算-通信重叠等。这些优化策略使我们在享受动态图调试便利性的同时，兼顾静态图的性能优势。实际操作中，需要在动态图的基础上做 3 处修改：
1. 使用 to_static 接口装饰顶层 model。
2. 调用 dist_model.train()  设置为训练模式。
3. 训练迭代中只需要调用 dist_model(inputs, labels) 即可，前向、反向、优化器步骤都封装在模型内部。

此外，可以通过配置 to_static 中的参数 [strategy](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distributed/Strategy_cn.html) ，来应用各类性能优化策略。

使用 to_static 进行自动并行动转静训练（以上文 DP2+MP4 的动态图训练脚本为基础），示例代码如下：

```python
# 运行方式： python -m paddle.distributed.launch --device=0,1,2,3,4,5,6,7 train.py
import numpy as np
import paddle
import paddle.distributed as dist
from paddle.io import BatchSampler, DataLoader, Dataset

mesh = dist.ProcessMesh([[0, 1, 2, 3], [4, 5, 6, 7]], dim_names=['dp', 'mp'])

class RandomDataset(Dataset):
    def __init__(self, seq_len, hidden, num_samples=100):
        super().__init__()
        self.seq_len = seq_len
        self.hidden = hidden
        self.num_samples = num_samples

    def __getitem__(self, index):
        input = np.random.uniform(size=[self.seq_len, self.hidden]).astype("float32")
        label = np.random.uniform(size=[self.seq_len, self.hidden]).astype('float32')
        return (input, label)

    def __len__(self):
        return self.num_samples

class MlpModel(paddle.nn.Layer):
    def __init__(self):
        super(MlpModel, self).__init__()
        self.w0 = self.create_parameter(shape=[1024, 4096])
        self.w1 = self.create_parameter(shape=[4096, 1024])

        self.w0 = dist.shard_tensor(self.w0, mesh, [dist.Replicate(), dist.Shard(1)])
        self.w1 = dist.shard_tensor(self.w1, mesh, [dist.Replicate(), dist.Shard(0)])

    def forward(self, x):
        y = paddle.matmul(x, self.w0)
        z = paddle.matmul(y, self.w1)
        return z

with paddle.LazyGuard():
    model = MlpModel()
for p in model.parameters():
    p.initialize()

dataset = RandomDataset(128, 1024)
sampler = BatchSampler(
    dataset,
    batch_size=4,
)
dataloader = DataLoader(
    dataset,
    batch_sampler=sampler,
)
dataloader = dist.shard_dataloader(dataloader, meshes=[mesh], shard_dims="dp")

opt = paddle.optimizer.AdamW(learning_rate=0.001, parameters=model.parameters())
opt = dist.shard_optimizer(opt)
loss_fn = paddle.nn.MSELoss()

# 动转静逻辑
strategy = dist.Strategy() # 通过配置 strategy 可以应用上各类性能优化策略
dist_model = dist.to_static(model, dataloader, loss_fn, opt, strategy=strategy)
dist_model.train()

for step, inputs in enumerate(dataloader):
    loss = dist_model(inputs)
    print('step: ', step, ' loss: ', loss)
```
strategy 中可配置的各类性能优化策略和用法如下：

<html>
<table>
<thead>
<tr>
<th align="center">分类</th>
<th align="center">优化策略</th>
<th>简介</th>
<th>用法</th>
</tr>
</thead>
<tbody>
<tr>
<td align="center" rowspan="2">重计算</td>
<td align="center">recompute</td>
<td>重计算技术是一种时间换空间的策略，在前向计算中舍弃一些中间变量不进行保存，反向计算时再重新计算这些变量。</td>
<td><code>strategy._recompute.enable = True</code><br>应用重计算策略时需要使用 recompute 接口对需要进行重计算的模型层进行装饰</td>
</tr>
<tr>
<td align="center">refined recompute</td>
<td>精细化重计算，指定重计算网络层中部分算子不进行重计算，最大化重计算的收益。</td>
<td><code>strategy._recompute.refined_ops_patterns = [{ "main_ops": ["matmul"], "num": -1, "pre_ops": ["multiply"], "suf_ops": ["relu"]}]</code><br>示例说明：定义了一组 pattern = pre_ops + other_ops + suf_ops，在匹配该 pattern 的网络中，main_ops 即 matmul 算子将不进行重计算。num 参数表示 pattern 匹配次数，只有前 num 个 pattern 的 main_ops 是不需要重计算的。</td>
</tr>
<tr>
<td align="center" rowspan="3">算子融合</td>
<td align="center">fused_linear_param_grad_add</td>
<td>将模型中的参数梯度计算和加法操作融合为一个算子，以减少计算开销。</td>
<td><code>strategy.fused_passes.fused_passes_list.append("fused_linear_param_grad_add_pass")</code></td>
</tr>
<tr>
<td align="center">fuse_linear</td>
<td>将 Linear 中的 matmul 和 add 算子融合为一个 fused_linear 算子，以减少计算开销和调度开销。</td>
<td><code>strategy.fused_passes.fused_passes_list.append("fused_gemm_epilogue_pass")</code></td>
</tr>
<tr>
<td align="center">fuse_allreduce_split_to_reducescatter</td>
<td>将 allreduce 通信和 split 切分操作融合为 reduce_scatter，以减少通信开销。</td>
<td><code>strategy.fused_passes.fused_passes_list.append("fuse_allreduce_split_to_reducescatter_pass")</code></td>
</tr>
<tr>
<td align="center" rowspan="3">参数切片并行优化</td>
<td align="center">tensor_fusion</td>
<td>将多个小的通信操作（如梯度或参数的传输）合并为一个大通信操作，以减少通信开销</td>
<td><code>strategy.sharding.enable_tensor_fusion = True</code></td>
</tr>
<tr>
<td align="center">通信计算 overlap</td>
<td>将分组切分并行中的计算与通信操作重叠，提升训练效率</td>
<td><code>strategy.sharding.enable_overlap = True</code></td>
</tr>
<tr>
<td align="center">release_gradients</td>
<td>释放梯度显存，以降低训练过程中的显存开销。</td>
<td><code>strategy.sharding.release_gradients = True</code></td>
</tr>
<tr>
<td align="center" rowspan="2">张量并行优化</td>
<td align="center">replace_with_c_embedding</td>
<td>将列切的 embedding 算子替换为行切的 c_embedding 算子，并自动插入通信</td>
<td><code>strategy._mp_optimization.replace_with_c_embedding = True</code></td>
</tr>
<tr>
<td align="center">replace_with_parallel_cross_entropy</td>
<td>将 cross_entropy_with_softmax 算子替换为带通信的算子 c_softmax_with_cross_entropy，以降低显存开销并加速计算。</td>
<td><code>strategy._mp_optimization.replace_with_parallel_cross_entropy = True</code></td>
</tr>
</tbody>
</table>
</html>


> 注：关于动转静技术的语法支持、调试方式、使用限制等，请参考文档：[动态图转静态图](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/jit/index_cn.html)。



## 四、模型保存与加载
模型训练完成后，通常需要使用模型保存（save）功能将其持久化到磁盘文件中，以便在后续的训练调优或推理部署时，可以加载（load）到内存中运行。对于参数规模较小的单卡模型，这些模型参数文件的读写操作通常在本地完成，这种方式相对简单，对系统资源的要求也较低。然而，在处理分布式大模型时，由于其参数规模巨大，模型的保存和加载不仅需要更多的存储容量，还对读写性能提出了更高的要求。

自动并行框架设计了一套动静统一的高效模型分片保存和加载方案，以解决单设备存储容量有限的问题，该功能同时支持以下特性：

1. **参数去重存储**：在数据并行等策略中，通信组内的参数完全一致，不需要冗余存储。
2. **动态分片调整**：当保存和加载阶段的并行策略不一致时（例如从 MP2 变为 MP4），支持参数在线重分片（reshard）。
3. **动静参数统一**：相同的网络结构在动态图和静态图的自动并行之间可以相互加载。
自动并行框架为用户提供了 [paddle.distributed.save_state_dict](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distributed/save_state_dict_cn.html) 和 [paddle.distributed.load_state_dict](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distributed/load_state_dict_cn.html)  API ，分别用于模型的保存和加载。在分布式自动并行模型运行完成后，如果需要保存模型和优化器的状态，只需添加 2 行代码即可实现：

```python
dist.save_state_dict(model.state_dict(), './ckpt/model')    # model 为模型名称
dist.save_state_dict(opt.state_dict(), './ckpt/opt')  # opt 为优化器名称
```
save_state_dict 方法接受 2 个参数，分别为需要保存的状态字典和用户指定的保存路径。以上文所提的 DP2+MP4 混合并行保存模型和优化器状态为例，在训练阶段结束后增加如下第 67 、68 行代码：

```python
# 运行方式： python -m paddle.distributed.launch --device=0,1,2,3,4,5,6,7 train.py
import numpy as np
import paddle
import paddle.distributed as dist
from paddle.io import BatchSampler, DataLoader, Dataset

mesh = dist.ProcessMesh([[0, 1, 2, 3], [4, 5, 6, 7]], dim_names=['dp', 'mp'])

class RandomDataset(Dataset):
    def __init__(self, seq_len, hidden, num_samples=100):
        super().__init__()
        self.seq_len = seq_len
        self.hidden = hidden
        self.num_samples = num_samples

    def __getitem__(self, index):
        input = np.random.uniform(size=[self.seq_len, self.hidden]).astype("float32")
        return input

    def __len__(self):
        return self.num_samples

class MlpModel(paddle.nn.Layer):
    def __init__(self):
        super(MlpModel, self).__init__()
        self.w0 = self.create_parameter(shape=[1024, 4096])
        self.w1 = self.create_parameter(shape=[4096, 1024])

        self.w0 = dist.shard_tensor(self.w0, mesh, [dist.Replicate(), dist.Shard(1)])
        self.w1 = dist.shard_tensor(self.w1, mesh, [dist.Replicate(), dist.Shard(0)])

    def forward(self, x):
        x = dist.shard_tensor(x, mesh, [dist.Shard(0), dist.Replicate()])
        y = paddle.matmul(x, self.w0)
        z = paddle.matmul(y, self.w1)
        return z

with paddle.LazyGuard():
    model = MlpModel()
for p in model.parameters():
    p.initialize()

dataset = RandomDataset(128, 1024)
sampler = BatchSampler(
    dataset,
    batch_size=4,
)
dataloader = DataLoader(
    dataset,
    batch_sampler=sampler,
)

opt = paddle.optimizer.AdamW(learning_rate=0.001, parameters=model.parameters())
opt = dist.shard_optimizer(opt)

for step, inputs in enumerate(dataloader):
    data = inputs
    logits = model(data)
    loss = paddle.mean(logits)
    loss.backward()
    opt.step()
    opt.clear_grad()

print(f"max_memory_reserved = {paddle.device.cuda.max_memory_reserved() / 1e6 : .2f} MB")

# 在模型运行结束后保存
dist.save_state_dict(model.state_dict(), './ckpt/model')
dist.save_state_dict(opt.state_dict(), './ckpt/opt')
```
程序运行完成时，将在 ‘./ckpt/model’ 和 ‘./ckpt/opt’ 目录下看到保存的模型和优化器状态文件。每个目录中分别包含以下文件：

* metadata 类型文件：0.metadata，用于记录所有设备内分片的分布式信息，系统可以利用这些记录快速还原出全局模型状态。其中前缀数字 0 为标识检查点（checkpoint）的版本号。
* distcp 类型文件：0_0.distcp、1_0.distcp 、2_0.distcp、3_0.distcp 、4_0.distcp、5_0.distcp 、6_0.distcp、7_0.distcp，用于存储模型参数和优化器分片的状态。文件名由两个数字组成，中间用下划线分隔。第一个数字表示文件当前所对应的设备 rank 编号，第二个数字为一个 unique_id，通常情况下为 0。
如果需要在上一次训练的基础上继续训练，或使用已训练完成的模型进行推理，可以使用 load_state_dict 方法重新加载已保存的模型和优化器状态。只需添加 2 行代码（第 57、58 行）即可实现：

```python
# 运行方式： python -m paddle.distributed.launch --device=0,1,2,3,4,5,6,7 train.py
import numpy as np
import paddle
import paddle.distributed as dist
from paddle.io import BatchSampler, DataLoader, Dataset

mesh = dist.ProcessMesh([[0, 1, 2, 3], [4, 5, 6, 7]], dim_names=['dp', 'mp'])

class RandomDataset(Dataset):
    def __init__(self, seq_len, hidden, num_samples=100):
        super().__init__()
        self.seq_len = seq_len
        self.hidden = hidden
        self.num_samples = num_samples

    def __getitem__(self, index):
        input = np.random.uniform(size=[self.seq_len, self.hidden]).astype("float32")
        return input

    def __len__(self):
        return self.num_samples

class MlpModel(paddle.nn.Layer):
    def __init__(self):
        super(MlpModel, self).__init__()
        self.w0 = self.create_parameter(shape=[1024, 4096])
        self.w1 = self.create_parameter(shape=[4096, 1024])

        self.w0 = dist.shard_tensor(self.w0, mesh, [dist.Replicate(), dist.Shard(1)])
        self.w1 = dist.shard_tensor(self.w1, mesh, [dist.Replicate(), dist.Shard(0)])

    def forward(self, x):
        x = dist.shard_tensor(x, mesh, [dist.Shard(0), dist.Replicate()])
        y = paddle.matmul(x, self.w0)
        z = paddle.matmul(y, self.w1)
        return z

with paddle.LazyGuard():
    model = MlpModel()
for p in model.parameters():
    p.initialize()

dataset = RandomDataset(128, 1024)
sampler = BatchSampler(
    dataset,
    batch_size=4,
)
dataloader = DataLoader(
    dataset,
    batch_sampler=sampler,
)

opt = paddle.optimizer.AdamW(learning_rate=0.001, parameters=model.parameters())
opt = dist.shard_optimizer(opt)

# 在模型训练阶段开始前加载
dist.load_state_dict(model.state_dict(), './ckpt/model')
dist.load_state_dict(opt.state_dict(), './ckpt/opt')

for step, inputs in enumerate(dataloader):
    data = inputs
    logits = model(data)
    loss = paddle.mean(logits)
    loss.backward()
    opt.step()
    opt.clear_grad()

print(f"max_memory_reserved = {paddle.device.cuda.max_memory_reserved() / 1e6 : .2f} MB")
```

<p id="pipeline"></p>

## 五、流水并行
流水并行将深度学习模型不同层放置于不同计算设备上，每个设备负责一个或多个模型层，以流水线的形式进行并行计算。根据模型层放置方式和执行方式的不同，流水并行有多种实现策略，它们在并行效率和显存消耗方面各有差异。飞桨自动并行目前支持以下三种流水并行技术：FThenB（Gpipe）、1F1B（Pipedream）和 VPP（Interleaved-pp）。

### 5.1 朴素流水并行实现
```python
# 运行方式： python train.py
import numpy as np
import paddle
import paddle.nn as nn
import paddle.distributed as dist
from paddle.io import BatchSampler, DataLoader, Dataset

mesh = dist.ProcessMesh([0, 1, 2, 3], dim_names=['pp'])

class RandomDataset(Dataset):
    def __init__(self, image_size, num_samples=100):
        super().__init__()
        self.image_size = image_size
        self.num_samples = num_samples

    def __getitem__(self, index):
        input = np.random.uniform(size=[self.image_size]).astype("float32")
        label = np.random.uniform(size=[10]).astype("float32")
        return input, label

    def __len__(self):
        return self.num_samples

class LinearModel(nn.Layer):
    def __init__(self, num_layers, image_size, class_num):
        super().__init__()
        self.num_layers = num_layers
        self.word_size = dist.get_world_size()
        self.image_size = image_size
        self.class_num = class_num

        self.linears = nn.LayerList()
        for i in range(num_layers):
            out_size = class_num if i == num_layers - 1 else image_size
            linear = nn.Linear(image_size, out_size, bias_attr=False)
            self.linears.append(linear)

        self.relus = nn.LayerList([nn.ReLU() for _ in range(num_layers + 1)])

    def forward(self, x):
        x.stop_gradient = False
        current_mesh = None
        out = self.relus[0](x)

        for i in range(self.num_layers):
            out = self.linears[i](out)
            out = self.relus[i+1](out)

        return paddle.cast(out, 'float32')

model = LinearModel(num_layers=8, image_size=4096, class_num=10)

dataset = RandomDataset(4096)
sampler = BatchSampler(
    dataset,
    batch_size=8,
    drop_last=True,
)
dataloader = DataLoader(
    dataset,
    batch_sampler=sampler,
)

opt = paddle.optimizer.AdamW(learning_rate=0.001, parameters=model.parameters())
loss_fn = nn.MSELoss()

for step, inputs in enumerate(dataloader):
    data, label = inputs[0], inputs[1]
    logits = model(data)

    loss = loss_fn(logits, label)
    loss.backward()
    opt.step()
    opt.clear_grad()

print(f"max_memory_reserved = {paddle.device.cuda.max_memory_reserved() / 1e6 : .2f} MB") # 1948.42 MB
```
上面是一个单卡的 Linear 模型代码，`LinearModel` 是一个由多个线性层和 ReLU 激活函数组成的简单多层感知器，其层数由 `num_layers` 动态定义，最后一层输出类别数，其他层保持输入维度，适用于分类任务。这里我们选择用一个 8 层的网络来演示。

假设我们有 4 张 GPU，那么可以将 8 层网络均匀分配到 4 张 GPU 上，每个 GPU 分配 2 层。在代码实现的时候分为 3 个步骤：

1. 标记网络参数：在`__init__` 方法中实现权重的分布式分配，将不同层的参数标记到不同设备上
2. 标记中间变量：`在 forward` 方法中将需要输出到其他设备的中间变量通过 `reshard` 操作标记跨设备切换
3. 标记数据：使用 `shard_dataloader`将输入数据和标签标记到正确的卡上，通常输入数据只在第一个设备上使用，label 只在最后一个设备上使用

```python
# 运行方式： python -m paddle.distributed.launch --device=0,1,2,3 train.py
import numpy as np
import paddle
import paddle.nn as nn
import paddle.distributed as dist
from paddle.io import BatchSampler, DataLoader, Dataset

mesh = dist.ProcessMesh([0, 1, 2, 3], dim_names=['pp'])
pp_degree = 4 # 定义流水并行度

class RandomDataset(Dataset):
    def __init__(self, image_size, num_samples=100):
        super().__init__()
        self.image_size = image_size
        self.num_samples = num_samples

    def __getitem__(self, index):
        input = np.random.uniform(size=[self.image_size]).astype("float32")
        label = np.random.uniform(size=[10]).astype("float32")
        return input, label

    def __len__(self):
        return self.num_samples

class LinearModel(nn.Layer):
    def __init__(self, num_layers, image_size, class_num):
        super().__init__()
        self.num_layers = num_layers
        self.image_size = image_size
        self.class_num = class_num
        self.num_layers_per_card = num_layers // pp_degree

        self.linears = nn.LayerList()
        for i in range(num_layers):
            out_size = class_num if i == num_layers - 1 else image_size
            linear = nn.Linear(image_size, out_size, bias_attr=False)

            # 标记网络参数
            linear.weight = dist.shard_tensor(
                linear.weight,
                self.get_pp_mesh(i),
                [dist.Replicate()],
            )

            self.linears.append(linear)

        self.relus = nn.LayerList([nn.ReLU() for _ in range(num_layers + 1)])

    # 获取 layer 对应的设备 mesh
    def get_pp_mesh(self, layer_index):
        # layer_index=0-7 对应的 mesh_idx 分别为 0,0,1,1,2,2,3,3
        mesh_idx = int(layer_index / (self.num_layers / pp_degree))
        return mesh[mesh_idx]

    def forward(self, x):
        x.stop_gradient = False
        out = self.relus[0](x)

        for i in range(self.num_layers):
            # 标记中间变量
            if i % self.num_layers_per_card == 0:
                out = dist.reshard(out, self.get_pp_mesh(i+1), [dist.Replicate()])

            out = self.linears[i](out)
            out = self.relus[i+1](out)

        return paddle.cast(out, 'float32')

model = LinearModel(num_layers=8, image_size=4096, class_num=10)

dataset = RandomDataset(4096)
sampler = BatchSampler(
    dataset,
    batch_size=8,
    drop_last=True,
)
dataloader = DataLoader(
    dataset,
    batch_sampler=sampler,
)

# 标记数据
dist_dataloader = dist.shard_dataloader(dataloader, shard_dims=[0, 0], meshes=[mesh[0], mesh[-1]])

opt = paddle.optimizer.AdamW(learning_rate=0.001, parameters=model.parameters())
loss_fn = nn.MSELoss()

for step, inputs in enumerate(dist_dataloader()):
    data, label = inputs[0], inputs[1]
    logits = model(data)

    loss = loss_fn(logits, label)
    loss.backward()
    opt.step()
    opt.clear_grad()

print(f"max_memory_reserved = {paddle.device.cuda.max_memory_reserved() / 1e6 : .2f} MB") # 1342.18 MB
```
这样我们就实现了 4 卡流水并行，其调度图如下所示：

<figure align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/docs/develop/docs/guides/paddle_v3_features/images/auto_parallel/v.png" width="70%"/>
</figure>

### 5.2 复杂流水并行策略
上面例子中的朴素流水并行存在一个问题：每个 GPU 在很大一部分时间内都处于空闲状态，这导致了计算资源的浪费。为了解决这一问题，飞桨自动并行在静态图模型下实现了 FthenB、1F1B 和 VPP 等更加高效的流水并行编排方式，以提升 GPU 的利用率。

在上文中已经介绍了如何通过`strategy` 来应用各类性能优化策略，流水并行调度策略同样也是通过`strategy`进行配置。假设我们想将上面的例子改成 1F1B 流水并行，模型组网和数据加载等代码不需改动，只需在动转静接口中配置以下并行策略，即可轻松实现高效并行：

* `schedule_mode` ：用于指定并行策略，目前支持 FThenB、1F1B 和 VPP。
* `pp_degree` ：表示流水并行度，即参与流水并行的 GPU 数量。例如，4 卡流水并行时设置为 4。
* `accumulate_steps` ：定义梯度累加的步数，也即 micro-batch 的数量。
如下配置表示启用流水并行，流水调度策略使用 1F1B，流水并行度为 4，micro-batch 数量为 8。

```python
strategy = dist.Strategy()
pipeline = strategy.pipeline
pipeline.enable = True
pipeline.schedule_mode = "1F1B"
pipeline.pp_degree = 4
pipeline.accumulate_steps = 8
```
完整代码如下：

```python
# 运行方式： python -m paddle.distributed.launch --device=0,1,2,3 train.py
import numpy as np
import paddle
import paddle.nn as nn
import paddle.distributed as dist
from paddle.io import BatchSampler, DataLoader, Dataset

mesh = dist.ProcessMesh([0, 1, 2, 3], dim_names=['pp'])
pp_degree = 4

class RandomDataset(Dataset):
    def __init__(self, image_size, num_samples=100):
        super().__init__()
        self.image_size = image_size
        self.num_samples = num_samples

    def __getitem__(self, index):
        input = np.random.uniform(size=[self.image_size]).astype("float32")
        label = np.random.uniform(size=[10]).astype("float32")
        return input, label

    def __len__(self):
        return self.num_samples

class LinearModel(nn.Layer):
    def __init__(self, num_layers, image_size, class_num):
        super().__init__()
        self.num_layers = num_layers
        self.image_size = image_size
        self.class_num = class_num
        self.num_layers_per_card = num_layers // pp_degree

        self.linears = nn.LayerList()
        for i in range(num_layers):
            out_size = class_num if i == num_layers - 1 else image_size
            linear = nn.Linear(image_size, out_size, bias_attr=False)

            # 标记网络参数
            linear.weight = dist.shard_tensor(
                linear.weight,
                self.get_pp_mesh(i),
                [dist.Replicate()],
            )

            self.linears.append(linear)

        self.relus = nn.LayerList([nn.ReLU() for _ in range(num_layers + 1)])

    # 获取 layer 对应的设备 mesh
    def get_pp_mesh(self, layer_index):
        # layer_index=0-7 对应的 mesh_idx 分别为 0,0,1,1,2,2,3,3
        mesh_idx = int(layer_index / (self.num_layers / pp_degree))
        return mesh[mesh_idx]

    def forward(self, x):
        x.stop_gradient = False
        out = self.relus[0](x)

        for i in range(self.num_layers):
            # 标记中间变量
            if i % self.num_layers_per_card == 0:
                out = dist.reshard(out, self.get_pp_mesh(i+1), [dist.Replicate()])

            out = self.linears[i](out)
            out = self.relus[i+1](out)

        return paddle.cast(out, 'float32')

model = LinearModel(num_layers=8, image_size=4096, class_num=10)

dataset = RandomDataset(4096)
sampler = BatchSampler(
    dataset,
    batch_size=8,
    drop_last=True,
)
dataloader = DataLoader(
    dataset,
    batch_sampler=sampler,
)

dist_dataloader = dist.shard_dataloader(dataloader, shard_dims=[0, 0], meshes=[mesh[0], mesh[-1]])

opt = paddle.optimizer.AdamW(learning_rate=0.001, parameters=model.parameters())
loss_fn = nn.MSELoss()

# 配置流水并行参数
strategy = dist.Strategy()
pipeline = strategy.pipeline
pipeline.enable = True
pipeline.schedule_mode = "1F1B"
pipeline.pp_degree = 4
pipeline.accumulate_steps = 8

model = dist.to_static(model, dist_dataloader, loss_fn, opt, strategy)
model.train()

for step, inputs in enumerate(dist_dataloader):
    loss = model(inputs)

print(f"max_memory_reserved = {paddle.device.cuda.max_memory_reserved() / 1e6 : .2f} MB") # 671.48 MB
```
各种流水并行调度图如下：
<figure align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/docs/develop/docs/guides/paddle_v3_features/images/auto_parallel/FThenB.png" width="70%"/>
<p>FThenB 调度图</p>
</figure>

<figure align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/docs/develop/docs/guides/paddle_v3_features/images/auto_parallel/1F1B.png" width="70%"/>
<p>1F1B 调度图</p>
</figure>

<figure align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/docs/develop/docs/guides/paddle_v3_features/images/auto_parallel/VPP.png" width="70%"/>
<p>VPP 调度图</p>
</figure>


> 注：
> 关于不同流水编排的更多细节可以参考相关论文（[FThenB](https://arxiv.org/abs/1811.06965), [1F1B](https://arxiv.org/abs/1806.03377), [VPP](https://arxiv.org/abs/2104.04473)）。
> 对于不同流水并行的底层实际调度方式，飞桨提供了一套方便的可视化工具，可参考[静态图流水并行时序图可视化工具](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/06_distributed_training/visual_pipeline_parallel_static_mode_cn.html)。


## 六、用户接口
下面表格汇总了自动并行提供的用户接口，你可以根据实际使用需要，查阅对应的接口文档：

|接口名|接口说明|
|-|-|
|[paddle.distributed.shard_tensor](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distributed/shard_tensor_cn.html#shard-tensor)|创建一个分布式张量|
|[paddle.distributed.reshard](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distributed/reshard_cn.html#reshard)|对分布式张量进行重切分|
|[paddle.distributed.shard_dataloader](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distributed/shard_dataloader_cn.html#cn-api-paddle-distributed-shard-dataloader)|将单卡视角的数据加载器转换成分布式视角，提供数据切分功能，常用于数据并行场景|
|[paddle.distributed.shard_optimizer](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distributed/shard_optimizer_cn.html#shard-optimizer)|将单卡的优化器转变为分布式的优化器，提供优化器切分功能，常用于参数切片并行场景|
|[paddle.distributed.shard_layer](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distributed/shard_layer_cn.html#shard-layer)|根据自定义切分方式将 Layer 中的输入、输出和模型参数转换为分布式张量|
|[paddle.distributed.to_static](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distributed/to_static_cn.html#to-static)|将自动并行动态图模型转换为静态图模型|
|[paddle.distributed.save_state_dict](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distributed/save_state_dict_cn.html#cn-api-paddle-distributed-save-state-dict)|对分布式模型进行保存|
|[paddle.distributed.load_state_dict](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distributed/load_state_dict_cn.html#cn-api-paddle-distributed-load-state-dict)|对分布式模型进行加载|
|[paddle.distributed.set_mesh](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distributed/set_mesh_cn.html#set-mesh)|设置全局 ProcessMesh|
|[paddle.distributed.get_mesh](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distributed/get_mesh_cn.html#get-mesh)|获取全局 ProcessMesh|
|[paddle.distributed.parallelize](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distributed/parallelize_cn.html#parallelize)|在模型组网外按照并行策略配置对模型和优化器进行分布式处理|

## 七、大模型套件使用
我们在 PaddleNLP 和 PaddleMIX 套件中提供了一些使用自动并行实现的模型案例，你可以点击以下链接遵照文档进行运行体验：

|文档|说明|
|-|-|
|[PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/auto_parallel/README.md)|介绍使用自动并行运行 GPT-3、Llama、Qwen 和 DeepSeek-V3 进行预训练、对齐训练和推理的方法。|
|[PaddleMIX](https://github.com/PaddlePaddle/PaddleMIX/blob/develop/paddlemix/examples/qwen2_vl/README.md)|介绍使用自动并行运行 Qwen2VL SFT、LoRA 和推理的方法。|

## 八、国产硬件运行
自动并行架构目前已支持昆仑芯，其它国产 AI 加速芯片的适配和优化正在进行中。你可以通过 PaddleNLP 大语言模型开发套件，在昆仑芯上[快速体验](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/auto_parallel/README.md#xpu-%E5%90%AF%E5%8A%A8%E9%A2%84%E8%AE%AD%E7%BB%83)飞桨的自动并行能力。

关于昆仑芯的安装和使用方式，可以参考：[硬件支持-昆仑芯 XPU](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/hardware_support/xpu/index_cn.html)。

## 九、更多学习资源
你可以从以下渠道，获取关于自动并行的更多学习资料：

* 飞桨官方小课堂：[https://space.bilibili.com/476867757/lists/4326139?type=season](https://space.bilibili.com/476867757/lists/4326139?type=season)
* 飞桨开源代码库自动并行目录：[https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle/distributed/auto_parallel](https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle/distributed/auto_parallel)
* 飞桨社区源码读书会：[https://github.com/PaddlePaddle/community/tree/master/pfcc/paddle-code-reading](https://github.com/PaddlePaddle/community/tree/master/pfcc/paddle-code-reading)

## 十、未来开发计划
自动并行作为飞桨 3.0 版本全新推出的分布式训练框架，为用户提供了一种学习和编程成本都更低的分布式实现方案。在后续的版本中，飞桨将持续迭代和优化自动并行的使用体验，构建更加高可用和高易用的自动并行架构。我们欢迎有大模型研发和创新需求的用户试用自动并行，并提出宝贵的意见和建议。你可以在飞桨 github 仓库中[提 issue](https://github.com/PaddlePaddle/Paddle/issues)反馈关于自动并行的问题和需求，也可以参考飞桨[贡献指南](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/index_cn.html)为自动并行项目贡献代码。

下面列出了一些我们正在进行，并可能在下个版本推出的工作：

* 基础能力完善
  - 分布式算子：简化分布式算子开发流程，包括探索更简单的切分推导规则编写方式，内置更多场景下常用算子的切分推导规则，完善自定义算子集成机制等。
  - 并行策略支持：开发 ContextParallel、ConvParallel 等更多并行策略，支持多模型结构混合并行，探索现有并行策略与 FP8、DeepEP 等大模型训练新技术的结合方式。
  - 切分标记语法：支持灵活标记更多的切分模式，包括非均衡切分、多维切分等。
  - 国产硬件支持：在更多国产 AI 加速芯片上适配自动并行，进行更深度的软硬件协同优化。
  - 自动并行推理：探索训推一体机制，使得自动并行组网的训练代码推理可复用，自动并行的静态图优化可以在推理中复用。
* 易用性提升
  - 流水并行：设计和开发更易用的流水并行接口，让用户能更简单和灵活地实现新的流水编排方式。
  - 全自动 API：基于 cost-model 为用户自动搜索和选择最优的并行策略，让用户可以不做任何切分标记，一键跑起分布式训练。
  - 调试体验：开发更多的调试工具，让用户能方便地调试自动并行程序。
* 性能优化
  - 动态图性能：在动态图上实现更多的优化策略，让用户在动态图模式下也能获得较优的性能体验。
  - AI 编译器：结合编译器技术构建图层自动并行+算子层编译器两层编译架构，探索分布式算子融合，实现更极致和通用的性能优化。

如果你对以上的研发内容感兴趣，或想在自动并行之上进行一些新的创新和研发，也可以加入飞桨开发者社区，一起建设和完善自动并行架构，共同定义理想的深度学习框架，欢迎访问社区 issue 区[置顶栏](https://github.com/PaddlePaddle/Paddle/issues)，参与飞桨启航计划、飞桨黑客松等丰富活动。关于飞桨开源社区的更多动态，欢迎关注[飞桨开源社区博客](https://pfcc.blog)！

你还可以通过以下方式直接联系到我们：

<style>
.image-group {
  display: flex;
  gap: 20px;
  justify-content: center;
  max-width: 70%;
  margin: 0 auto;
}

.image-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  width: 48%;
}

.image-title {
  font-family: Arial, sans-serif;
  font-size: 14px;
  color: #333;
  margin-top: 8px;
  text-align: center;
  max-width: 90%;
}

@media (max-width: 768px) {
  .image-group {
    flex-direction: column;
    align-items: center;
  }

  .image-container {
    width: 90%;
  }
}
</style>

<div class="image-group">
  <div class="image-container">
    <img src="https://raw.githubusercontent.com/PaddlePaddle/docs/develop/docs/guides/paddle_v3_features/images/auto_parallel/wechat-group.png" style="width:100%; height: auto; border-radius: 8px;">
    <div class="image-title">微信交流群</div>
  </div>

  <div class="image-container">
    <img src="https://raw.githubusercontent.com/PaddlePaddle/docs/develop/docs/guides/paddle_v3_features/images/auto_parallel/hi-group.png" style="width:100%; height: auto; border-radius: 8px;">
    <div class="image-title">如流交流群</div>
  </div>
</div>
