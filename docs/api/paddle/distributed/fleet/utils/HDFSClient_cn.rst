.. _cn_api_distributed_fleet_utils_fs_HDFSClient:

HDFSClient
-------------------------------

.. py:class:: paddle.distributed.fleet.utils.HDFSClient
一个HADOOP文件系统工具类。

参数
::::::::::::

    - **hadoop_home** (str)：HADOOP HOME地址。
    - **configs** (dict): HADOOP文件系统配置。需包含 `fs.default.name` 和 `hadoop.job.ugi` 这两个字段。

代码示例
::::::::::::

.. code-block:: python

    from paddle.distributed.fleet.utils import HDFSClient
    hadoop_home = "/home/client/hadoop-client/hadoop/"

    configs = {
        "fs.default.name": "hdfs://xxx.hadoop.com:54310",
        "hadoop.job.ugi": "hello,hello123"
    }

    client = HDFSClient(hadoop_home, configs)
    client.ls_dir("hdfs:/test_hdfs_client")

方法
::::::::::::
ls_dir(fs_path)
'''''''''
列出 `fs_path` 路径下所有的文件和子目录。

**参数**

    - **fs_path** (str): HADOOP文件路径。

**返回**

    - Tuple，一个包含所有子目录和文件名的2-Tuple，格式形如：([subdirname1, subdirname1, ...], [filename1, filename2, ...])。

**代码示例**

.. code-block:: python

    from paddle.distributed.fleet.utils import HDFSClient

    hadoop_home = "/home/client/hadoop-client/hadoop/"
    configs = {
        "fs.default.name": "hdfs://xxx.hadoop.com:54310",
        "hadoop.job.ugi": "hello,hello123"
    }

    client = HDFSClient(hadoop_home, configs)
    subdirs, files = client.ls_dir("hdfs:/test_hdfs_client")

mkdirs(fs_path)
'''''''''
创建一个目录。

**参数**

    - **fs_path** (str): HADOOP文件路径。

**代码示例**

.. code-block:: python

    from paddle.distributed.fleet.utils import HDFSClient

    hadoop_home = "/home/client/hadoop-client/hadoop/"
    configs = {
        "fs.default.name": "hdfs://xxx.hadoop.com:54310",
        "hadoop.job.ugi": "hello,hello123"
    }

    client = HDFSClient(hadoop_home, configs)
    client.mkdirs("hdfs:/test_hdfs_client")

delete(fs_path)
'''''''''
删除HADOOP文件（或目录）。

**参数**

    - **fs_path** (str): HADOOP文件路径。

**代码示例**

.. code-block:: python

    from paddle.distributed.fleet.utils import HDFSClient

    hadoop_home = "/home/client/hadoop-client/hadoop/"
    configs = {
        "fs.default.name": "hdfs://xxx.hadoop.com:54310",
        "hadoop.job.ugi": "hello,hello123"
    }

    client = HDFSClient(hadoop_home, configs)
    client.delete("hdfs:/test_hdfs_client")

is_file(fs_path)
'''''''''
判断当前路径是否是一个文件。

**参数**

    - **fs_path** (str): HADOOP文件路径。

**返回**

    - Bool：若当前路径存在且是一个文件，返回 `True`，反之则返回 `False` 。

**代码示例**

.. code-block:: python

    from paddle.distributed.fleet.utils import HDFSClient

    hadoop_home = "/home/client/hadoop-client/hadoop/"
    configs = {
        "fs.default.name": "hdfs://xxx.hadoop.com:54310",
        "hadoop.job.ugi": "hello,hello123"
    }

    client = HDFSClient(hadoop_home, configs)
    ret = client.is_file("hdfs:/test_hdfs_client")

is_dir(fs_path)
'''''''''
判断当前路径是否是一个目录。

**参数**

    - **fs_path** (str): HADOOP文件路径。

**返回**

    - Bool：若当前路径存在且是一个目录，返回 `True`，反之则返回 `False` 。

**代码示例**

.. code-block:: python

    from paddle.distributed.fleet.utils import HDFSClient

    hadoop_home = "/home/client/hadoop-client/hadoop/"
    configs = {
        "fs.default.name": "hdfs://xxx.hadoop.com:54310",
        "hadoop.job.ugi": "hello,hello123"
    }

    client = HDFSClient(hadoop_home, configs)
    ret = client.is_file("hdfs:/test_hdfs_client")

is_exist(fs_path)
'''''''''
判断当前路径是否存在。

**参数**

    - **fs_path** (str): HADOOP文件路径。

**返回**

    - Bool：若当前路径存在返回 `True`，反之则返回 `False` 。

**代码示例**

.. code-block:: python

    from paddle.distributed.fleet.utils import HDFSClient

    hadoop_home = "/home/client/hadoop-client/hadoop/"
    configs = {
        "fs.default.name": "hdfs://xxx.hadoop.com:54310",
        "hadoop.job.ugi": "hello,hello123"
    }

    client = HDFSClient(hadoop_home, configs)
    ret = client.is_exist("hdfs:/test_hdfs_client")

upload(local_path, fs_path)
'''''''''
上传本地文件至HADOOP文件系统。

**参数**

    - **local_path** (str)：本地文件路径。
    - **fs_path** (str): HADOOP文件路径。

**代码示例**

.. code-block:: python

    from paddle.distributed.fleet.utils import HDFSClient

    hadoop_home = "/home/client/hadoop-client/hadoop/"
    configs = {
        "fs.default.name": "hdfs://xxx.hadoop.com:54310",
        "hadoop.job.ugi": "hello,hello123"
    }

    client = HDFSClient(hadoop_home, configs)
    client.upload("test_hdfs_client", "hdfs:/test_hdfs_client")

download(fs_path, local_path)
'''''''''
下载HADOOP文件至本地文件系统。

**参数**

    - **local_path** (str)：本地文件路径。
    - **fs_path** (str): HADOOP文件路径。

**代码示例**

.. code-block:: python

    from paddle.distributed.fleet.utils import HDFSClient

    hadoop_home = "/home/client/hadoop-client/hadoop/"
    configs = {
        "fs.default.name": "hdfs://xxx.hadoop.com:54310",
        "hadoop.job.ugi": "hello,hello123"
    }

    client = HDFSClient(hadoop_home, configs)
    client.download("hdfs:/test_hdfs_client", "./")


touch(fs_path, exist_ok=True)
'''''''''
创建一个HADOOP文件。

**参数**

    - **fs_path** (str): HADOOP文件路径。
    - **exist_ok** (bool)：路径已存在时程序是否报错。若 `exist_ok = True`，则直接返回，反之则抛出文件存在的异常，默认不抛出异常。

**代码示例**

.. code-block:: python

    from paddle.distributed.fleet.utils import HDFSClient

    hadoop_home = "/home/client/hadoop-client/hadoop/"
    configs = {
        "fs.default.name": "hdfs://xxx.hadoop.com:54310",
        "hadoop.job.ugi": "hello,hello123"
    }

    client = HDFSClient(hadoop_home, configs)
    client.touch("hdfs:/test_hdfs_client")

mv(fs_src_path, fs_dst_path, overwrite=False)
'''''''''
HADOOP系统文件移动。

**参数**

    - **fs_src_path** (str)：移动前源文件路径名。
    - **fs_dst_path** (str)：移动后目标文件路径名。
    - **overwrite** (bool)：若目标文件已存在，是否删除进行重写，默认不重写并抛出异常。
 
**代码示例**

.. code-block:: python

    from paddle.distributed.fleet.utils import HDFSClient

    hadoop_home = "/home/client/hadoop-client/hadoop/"
    configs = {
        "fs.default.name": "hdfs://xxx.hadoop.com:54310",
        "hadoop.job.ugi": "hello,hello123"
    }

    client = HDFSClient(hadoop_home, configs)
    client.mv("hdfs:/test_hdfs_client", "hdfs:/test_hdfs_client2")

list_dirs(fs_path)
'''''''''
列出HADOOP文件路径下所有的子目录。

**参数**

    - **fs_path** (str): HADOOP文件路径。

**返回**

    - List：该路径下所有的子目录名。

**代码示例**

.. code-block:: python

    from paddle.distributed.fleet.utils import HDFSClient

    hadoop_home = "/home/client/hadoop-client/hadoop/"
    configs = {
        "fs.default.name": "hdfs://xxx.hadoop.com:54310",
        "hadoop.job.ugi": "hello,hello123"
    }

    client = HDFSClient(hadoop_home, configs)
    subdirs = client.list_dirs("hdfs:/test_hdfs_client")



