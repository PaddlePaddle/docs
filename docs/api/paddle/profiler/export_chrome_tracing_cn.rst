.. _cn_api_profiler_export_chrome_tracing:

export_chrome_tracing
---------------------

.. py:function:: paddle.profiler.export_chrome_tracing(dir_name: str, worker_name: Optional[str]=None)

该接口用于生成将性能数据保存到google chrome tracing文件的回调函数。

参数:
    - **dir_name** (str) - 性能数据导出所保存到的文件夹路径。
    - **worker_name** (str, 可选) - 性能数据导出所保存到的文件名前缀，默认是[hostname]_[pid]。

返回: 回调函数（callable), 该函数会接收一个参数prof(Profiler对象），调用prof的export方法保存采集到的性能数据到chrome tracing文件。
