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

    place 表示 op 或者 variable 所处的位置，它可以通过一个字符串来描述：

    ```text
    place ::= /("role" : worker/ps : [0-9]*)/(cpu/gpu : [0-9]*)

    valide value:
    place = "/role:worker/gpu:8"    // running on GPU with 8 cards
    place = "/role:ps/cpu:24"       // running on CPU with 24 cores
    ```

    - 设备: CPU 或者 GPU 设备.
    - 节点角色: 可以是 worker 或者 pserver.

1. 处于不同 place 的 op 如何通信？

    相邻的两个 op 可能处于不同的设备或者不同的节点上：
     - op 在不同设备上

        假设我们有两个 Operator: `OP1, OP2`, 他们属于同一个 worker 节点的不同设备 `OP1(/role:worker/gpu), OP2(/role:worker/cpu)`
        `out_var` 是 OP1 的输出并且是 OP2 的输入，那么很显然 OP2 是无法直接使用 OP1 的输出结果的:

        ``` text
        OP1(/role:worker/gpu) -> var -> OP2(/role:worker/cpu)
        ```

        解决方法也比较容易, 只要在OP1 和 OP2 之间插入内存拷贝的 OP 即可

        ``` text
        OP1(/role:worker/gpu) -> var -> MemCpyD2H(var) -> OP2(/role:worker/cpu)
        ```

    - op 在不同的节点上

        同样假设我们有两个 Operator：`OP1, OP2`, 他们分别属于 worker 节点和 ps 节点 `OP1(/role:worker/gpu), OP2(/role:ps/cpu)`,
        `out_var` 是 OP1 的输出并且是 OP2 的输入，他们属于不同节点的不同设备:

        ``` text
        OP1(/role:worker/gpu) -> var -> OP2(/role:ps/cpu)
        ```

        这时我们需要在 OP1 和 OP2 之间插入 RPC 的 OP:

        ``` text
        OP1(/role:worker/gpu) -> var -> send(var) ... listen_and_serv(var) -> OP2(/role:ps/cpu)
        ```
