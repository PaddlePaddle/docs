.. _cn_api_distributed_release_gloo:

release_gloo
-------------------------------

.. py:function:: paddle.distributed.release_gloo()

释放当前并行环境的 gloo 上下文。

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

        def test_release_gloo(id, rank_num, server_endpoint):
            paddle.distributed.init_gloo_parallel_env(
                id, rank_num, server_endpoint)
            paddle.distributed.barrier_func()
            paddle.distributed.release_gloo(id)

        def test_release_gloo_with_multiprocess(num_of_ranks):
            jobs = []
            server_endpoint = "127.0.0.1:%s" % (find_free_port())
            for id in range(num_of_ranks):
                p = multiprocessing.Process(
                    target=test_release_gloo,
                    args=(id, num_of_ranks, server_endpoint))
                jobs.append(p)
                p.start()
            for proc in jobs:
                proc.join()

        if __name__ == '__main__':
            # Arg: number of ranks (processes)
            test_release_gloo_with_multiprocess(2)
