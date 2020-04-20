import pydoop.hdfs as hdfs

path = "/tmp/yongxi/tfoutput/mnist_model/xx.txt"

with hdfs.open(path, mode='wt', user='profile') as f:
    print("fuck you hdfs!!", file=f)
