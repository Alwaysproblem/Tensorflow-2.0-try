set -ex

# set environment variables (if not already done)
# conda env
CONDAENV=tf2dis

# /usr/hdp/2.5.6.0-40/hadoop/lib/native
# /usr/hdp/2.5.6.0-40/hadoop/libexec/hadoop-config.sh

# export HADOOP_HDFS_HOME=/usr/hdp/2.5.6.0-40/hadoop-hdfs
export HADOOP_HOME=/usr/hdp/2.5.6.0-40/hadoop
export CLASSPATH=$(hadoop classpath --glob)
# export LD_LIBRARY_PATH=${PATH}
export LD_LIBRARY_PATH=$(hadoop classpath):${JAVA_HOME}/jre/lib/amd64/server:/home/sdev/yongxi/env/tfhdfs/lib
export PYSPARK_PYTHON="./${CONDAENV}_zip/${CONDAENV}/bin/python"
export QUEUE=adx
export SPARK_HOME=/home/sdev/yongxi/spark-2.4.4-bin-hadoop2.7

# set paths to libjvm.so, libhdfs.so, and libcuda*.so
export LIB_HDFS=./${CONDAENV}_zip/env/tfhdfs/lib                                     # path to libhdfs.so, for TF acccess to HDFS
# already upto hdfs:///user-profile/yongxi/spark/env/tfhdfs/lib
# export LIB_JVM=hdfs:///user-profile/yongxi/spark/env/tfjvm                           # path to libjvm.so
# # set paths to libjvm.so, libhdfs.so, and libcuda*.so
# export LIB_HDFS=$HADOOP_PREFIX/lib/native/Linux-amd64-64
# # already upto hdfs:///user-profile/yongxi/spark/env/tfjvm
# export LIB_JVM=$JAVA_HOME/jre/lib/amd64/server
export LIB_JVM=./${CONDAENV}_zip/env/tfjvm

# on the cluster the path for lihdfs.so and libjvm.so
# /usr/hdp/2.5.6.0-40/usr/lib/libhdfs.so
# /usr/lib/ams-hbase/lib/hadoop-native/libhdfs.so

export HADOOP_USER_NAME=profile
export CLASSPATH=$(hadoop classpath --glob)

# python