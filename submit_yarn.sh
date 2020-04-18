set -ex

# set environment variables (if not already done)
# conda env
CONDAENV=tf2dis
export HADOOP_HDFS_HOME=/usr/hdp/2.5.6.0-40/hadoop-hdfs
export LD_LIBRARY_PATH=${PATH}:${HADOOP_HDFS_HOME}:${JAVA_HOME}/jre/lib/amd64/server
export PYSPARK_PYTHON="./${CONDAENV}_zip/${CONDAENV}/bin/python"
export QUEUE=adx
export SPARK_HOME=/home/sdev/yongxi/spark-2.4.4-bin-hadoop2.7

# set paths to libjvm.so, libhdfs.so, and libcuda*.so
export LIB_HDFS=/usr/lib/                                                # path to libhdfs.so, for TF acccess to HDFS
export LIB_JVM=$JAVA_HOME/jre/lib/amd64/server                           # path to libjvm.so

# on the cluster the path for lihdfs.so and libjvm.so
# /usr/hdp/2.5.6.0-40/usr/lib/libhdfs.so
# /usr/lib/ams-hbase/lib/hadoop-native/libhdfs.so

export HADOOP_USER_NAME=profile

export CLASSPATH=$(hadoop classpath --glob)

# jar Package on the air
TFCONNECTOR=hdfs:///user-profile/yongxi/spark/jars/spark-tensorflow-connector_2.11-1.15.0.jar
TFHADOOP=hdfs:///user-profile/yongxi/spark/jars/tensorflow-hadoop-1.15.0.jar


# spark configuration
SPARK_WORKER_INSTANCES=5
EXECUTOR_MEMORY=2G

# Train configuration
EPOCHS=2

# Input and output and not "hdfs://" pre-ffix
# and must obtain the write permission all the way of the path.
# because tensorflow will be create recursively files and paths.
export INPUT_DATA=/user-profile/yongxi/spark/input/mnist/csv/train
export MODEL_DIR=/tmp/yongxi/tfoutput/mnist_model
export EXPORT_DIR=/tmp/yongxi/tfoutput/mnist_export

sudo -u ${HADOOP_USER_NAME} hadoop fs -rm -r -f -skipTrash ${MODEL_DIR}/*
sudo -u ${HADOOP_USER_NAME} hadoop fs -rm -r -f -skipTrash ${EXPORT_DIR}/*

sudo -u ${HADOOP_USER_NAME} ${SPARK_HOME}/bin/spark-submit \
                    --master yarn \
                    --deploy-mode cluster \
                    --queue ${QUEUE} \
                    --num-executors ${SPARK_WORKER_INSTANCES} \
                    --executor-memory ${EXECUTOR_MEMORY} \
                    --conf spark.dynamicAllocation.enabled=false \
                    --conf spark.yarn.maxAppAttempts=1 \
                    --conf "spark.yarn.appMasterEnv.PYSPARK_PYTHON=./${CONDAENV}_zip/${CONDAENV}/bin/python" \
                    --conf spark.executorEnv.LD_LIBRARY_PATH=$LIB_JVM:$LIB_HDFS \
                    --conf spark.network.timeout=60000s \
                    --conf spark.executorEnv.CLASSPATH=${CLASSPATH} \
                    --conf spark.executorEnv.HADOOP_USER_NAME=${HADOOP_USER_NAME} \
                    --archives "../${CONDAENV}.zip#${CONDAENV}_zip" \
                    --jars ${TFCONNECTOR},${TFHADOOP} \
                    mnist_spark.py \
                        --cluster_size ${SPARK_WORKER_INSTANCES} \
                        --epochs ${EPOCHS} \
                        --images_labels ${INPUT_DATA} \
                        --model_dir ${MODEL_DIR} \
                        --export_dir ${EXPORT_DIR}
                    # try_spark.py \
