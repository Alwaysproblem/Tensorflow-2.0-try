import pydoop.hdfs as hdfs

path = "/tmp/yongxi/tfoutput/mnist_model/xx.txt"
# hdfs.dump('hello, world', path)

with hdfs.open(path, mode='w', user='profile') as f:
    print("fuck you hdfs!!", file=f)
