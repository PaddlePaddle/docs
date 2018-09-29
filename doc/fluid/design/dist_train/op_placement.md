# Operator Placement Design

## 背景

大规模的并行训练通常分为两种场景：数据并行和模型并行，有关这两种场景的说明可以参考[分布式训练基本思想](http://paddlepaddle.org/documentation/docs/zh/0.15.0/user_guides/howto/training/cluster_howto.html#id1)

目前 Fluid 中实现并行训练的思路是通过 `DistributeTranspiler` 模块将用户配置的单机 `program` 拆分为多个 `sub-programs`，每个
`sub-program` 可以运行在集群中的一个节点上: `parameter server` 或者 `worker`. 但这样的做法有以下的局限性：
1. `parameter server` 节点上只可以运行做参数优化的 operator:

    `DistributeTranspiler` 通过识别 operator 的 role, 而不是 operator 的 place 决定哪些 operator 应该放在 worker，哪些 operator 应该放在 parameter server 上。

1. 无法实现模型并行

    Fluid 中目前无法指定哪些 Layer 运行在哪个节点或者哪个设备上。并且目前的 `ParallelExecutor` 也只支持数据并行。

本文将会介绍一种允许用户在配置网络时指定 operator 的 place 的方法，来解决以上问题:
1. `DistributedTranspiler` 可以通过识别 operator 的 place，将 `place=ps` 的 operator 放在 parameter server 上，这样用户可以自由指定哪些 operator 需要运行在 parameter server 上，设计上更加通用。
1. 通过对 operator 指定 place 的方式，用户也可以很容易配置出一个模型并行的网络.

样例代码如下：

``` python
with fluid.place_guard(place="/node:worker/gpu:0-7"):
    x = fluid.layers.data(name='x', shape=[12, 12], type='float32')
    fc1 = fluid.layers.fc(input=x, size=128)
    ...
    append_backward()

with fluid.place_guard(place="/node:worker/cpu"):
    opt = fluid.optimizer.SGD(learning_rate=0.1)
    opt.minimize(avg_cost)

```

在上述代码中用户可以将前向以及反向的计算 operators 放在 GPU 设备上进行计算，而将执行参数更新的 operators 放在 CPU 设备上运行。

## 实现

### place 能够表示什么？

用户可以用一个字符串来表示 operator 的 place，字符串的格式定义如下：

``` text
place ::= "/node:$NODE_NAME:($NODE_ID)/$DEVICE:$(DEVICE_ID)"

valide value:
place = "/node:worker/gpu"
place = "/node:worker/gpu:2"
place = "/node:ps/cpu"
```

- `NODE_NAME`: 用来指定 operator 运行的节点, 可以是 `worker` 或者 `ps`。
- `NODE_ID`: 在模型并行的场景下，用户可以通过指定 operator 在哪个 worker 节点上运行。
- `DEVICE`: 目前 Fluid 支持 CPU 和 GPU 两种运算设备。
- `DEVICE_ID`: 在模型并行场景下，用户可以指定 operator 在哪个设备上运行。

### 如何连接两个属于不同 place 的 operator ？

在 `Fluid::SSAGraph` 中，两个 operator 通过 dependency var 产生依赖关系:

```text
OP1 -> dep_var -> OP2
```

在 `Fluid::ParallelExecutor` 中，`OP2` 只有在其 input var: `dep_var` 变为 ready 之后才会被执行，这里的 ready 指的是:

1. `OP1` 执行完毕，正确的输出 `dep_var`; 并且
1. `dep_var` 和 `OP2` 在同一个设备上。  

当我们制定了 operator 的 place 之后，有依赖关系的两个 operator 可能会属于不同的 place, 这会造成上述第二个条件无法满足从而使 `dep_var` 无法达到 ready 的状态, 我们的解决方法也比较简单：在不同 place 的 operator 之间插入相应的通信 operator 即可:

1. `OP1` 和 `OP2` 属于同一个节点的不同设备:

    假设我们指定 `OP1` 在 worker 节点的 GPU 上运行，`OP2` 在 worker 节点的 CPU 上运行:

    ``` text
    OP1(/node:worker/gpu) -> dep_var -> OP2("/node:worker/cpu)
    ```

    我们需要在 `OP1` 和 `OP2` 之间插入数据拷贝的 operator：

    ``` text
    # GPU device
    OP1(/node:worker/gpu) -> dep_var(/node:worker/gpu) -> MemCpyD2H(dep_var)

    # CPU device
    -> dep_var(/node:worker/cpu) -> OP2(/node:worker/cpu)
    ```
1. `OP1` 和 `OP2` 属于不同的节点:

    `OP1` 和 `OP2` 也能属于不同的计算节点：

    ``` text
    OP1("/node:worker:0/gpu") -> dep_var -> OP2("/node:worker:1/gpu")
    ```

    这时我们需要在 `OP1` 和 `OP2` 之间插入 RPC 通信的 operator:

    ``` text
    # worker 0
    OP1(/node:worker:0/gpu) -> dep_var(/node:worker:0/gpu) -> send(dep_var)

    ... RPC CHANNEL ...

    # worker 1 
    listen_and_serv(dep_var) -> dep_var(/node:worker:1/gpu) -> OP2(/node:worker:1/gpu)
    ```
