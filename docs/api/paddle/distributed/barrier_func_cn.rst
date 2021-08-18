.. _cn_api_distributed_barrier_func:

barrier_func
-------------------------------


.. py:function:: paddle.distributed.barrier_func()
使用初始化的 gloo 上下文直接调用基于 gloo 封装的 barrier 函数；区别于使用组网方式调用。

参数
:::::::::
无

返回
:::::::::
无

代码示例
:::::::::
.. code-block:: python

        import paddle
        import multiprocessing
        from contextlib import closing
        import socket

        port_set = set()

        def find_free_port():
            def _free_port():
                with closing(socket.socket(socket.AF_INET,
                    socket.SOCK_STREAM)) as s:
                    s.bind(('', 0))
                    return s.getsockname()[1]
            while True:
                port = _free_port()
                if port not in port_set:
                    port_set.add(port)
                    return port

        def test_barrier_func(id, rank_num, server_endpoint):
            paddle.distributed.init_gloo_parallel_env(
                id, rank_num, server_endpoint)
            paddle.distributed.barrier_func()

        def test_barrier_with_multiprocess(num_of_ranks):
            jobs = []
            server_endpoint = "127.0.0.1:%s" % (find_free_port())
            for id in range(num_of_ranks):
                p = multiprocessing.Process(
                    target=test_barrier_func,
                    args=(id, num_of_ranks, server_endpoint))
                jobs.append(p)
                p.start()
            for proc in jobs:
                proc.join()

        if __name__ == '__main__':
            # Arg: number of ranks (processes)
            test_barrier_with_multiprocess(2)
