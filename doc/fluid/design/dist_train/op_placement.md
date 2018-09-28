# Operator Placement Design

## 背景

大规模的并行训练通常分为两种场景：数据并行和模型并行，有关这两种场景的说明可以参考链接：
目前 Fluid 只支持多机多设备的数据并行训练，本文中将会介绍一种方法，用户可以使用 `fluid.place_guard()` 来指定 Layer 所在的 place 从而实现模型并行。样例代码如下：

``` python
with fluid.place_guard(place="/role:worker/gpu:0-7"):
    x = fluid.layers.data(name='x', shape=[12, 12], type='float32')
    fc1 = fluid.layers.fc(input=x, size=128)
    ...

with fluid.place_guard(place="/role:worker/cpu"):
    opt = fluid.optimizer.SGD(learning_rate=0.1)
    opt.minimize(avg_cost)

```

## 问题及解决方法

1. place 是什么？

    place 用来表示 op 所处的设备以及节点，它可以通过一个字符串来描述：

    ```text
    place ::= /("role" : worker/ps : [0-9]*)/(cpu/gpu : [0-9]*)

    valide value:
    place = "/role:worker/gpu:8"    // running on GPU with 8 cards
    place = "/role:ps/cpu:24"       // running on CPU with 24 cores
    ```

    - 设备: CPU 或者 GPU 设备.
    - 节点角色: 可以是 worker 或者 pserver.

1. 如何处理两个处于不同 place 的计算 op ？

    在 Fluid::ParallelExecutor 中，一个计算 op 只有在 input var 变为 ready 之后才会被执行，这里的 ready 包含两层含义：
    - 依赖的 op 执行完毕
    - input var 和 op 所在的 place 相同

    当我们指定了 op 的 place 之后，相邻的两个计算 op 可能会处于不同的 place，这会造成 input var 永远不会 ready,
    此时我们需要在两个 op 之间插入通信 op， 确保 input var 能够达到 ready 的状态:
    - 计算 op 在不同设备:

        假设我们有两个 op: `OP1, OP2`, 他们属于同一个 worker 节点的不同设备 `OP1(/role:worker/gpu), OP2(/role:worker/cpu)`
        `dep_var` 是 OP1 的输出并且是 OP2 的输入，那么很显然 OP2 是无法直接使用 OP1 的输出结果的:

        ``` text
        OP1(/role:worker/gpu) -> dep_var -> OP2(/role:worker/cpu)
        ```

        解决方法也比较容易, 只要在OP1 和 OP2 之间插入数据拷贝的通信 op:

        ``` text
        OP1(/role:worker/gpu) -> dep_var -> MemCpyD2H(dep_var) -> OP2(/role:worker/cpu)
        ```

    - 计算 op 在不同节点

        同样假设我们有两个 op：`OP1, OP2`, 他们分别属于 worker 节点和 ps 节点 `OP1(/role:worker/gpu), OP2(/role:ps/cpu)`,
        `out_var` 是 OP1 的输出并且是 OP2 的输入，他们属于不同节点的不同设备:

        ``` text
        OP1(/role:worker/gpu) -> dep_var -> OP2(/role:ps/cpu)
        ```

        这时我们需要在 OP1 和 OP2 之间插入 RPC 的通信 op:

        ``` text
        OP1(/role:worker/gpu) -> dep_var -> send(dep_var) ... listen_and_serv(dep_var) -> OP2(/role:ps/cpu)
        ```
