import pydoop.hdfs as hdfs

path = "/tmp/yongxi/tfoutput/mnist_model/xx.txt"
hdfs.dump('hello, world', path)
