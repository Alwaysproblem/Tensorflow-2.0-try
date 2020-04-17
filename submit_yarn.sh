set -ex
# set environment variables (if not already done)
# export PYTHON_ROOT=./Python
export LD_LIBRARY_PATH=${PATH}
# export PYSPARK_PYTHON=${PYTHON_ROOT}/bin/python
# export SPARK_YARN_USER_ENV="PYSPARK_PYTHON=Python/bin/python"
# export PATH=${PYTHON_ROOT}/bin/:$PATH
export PYSPARK_PYTHON="./${CONDAENV}_zip/${CONDAENV}/bin/python"
export QUEUE=adx
export SPARK_HOME=/home/sdev/yongxi/spark-2.4.4-bin-hadoop2.7

# set paths to libjvm.so, libhdfs.so, and libcuda*.so
#export LIB_HDFS=/opt/cloudera/parcels/CDH/lib64                         # for CDH (per @wangyum)
export LIB_HDFS=/usr/lib/ams-hbase/lib/hadoop-native/                   # path to libhdfs.so, for TF acccess to HDFS
export LIB_JVM=$JAVA_HOME/jre/lib/amd64/server                           # path to libjvm.so
# export LIB_CUDA=/usr/local/cuda-7.5/lib64                              # for GPUs only

# on the cluster the path for lihdfs.so and libjvm.so
# /usr/hdp/2.5.6.0-40/usr/lib/libhdfs.so
# /usr/lib/ams-hbase/lib/hadoop-native/libhdfs.so
export HADOOP_HDFS_HOME=/usr/hdp/2.5.6.0-40/hadoop-hdfs

# jar Package on the air
TFCONNECTOR=hdfs:///user-profile/yongxi/spark/jars/spark-tensorflow-connector_2.11-1.15.0.jar
TFHADOOP=hdfs:///user-profile/yongxi/spark/jars/tensorflow-hadoop-1.15.0.jar

# conda env
CONDAENV=tf2dis

# spark configuration
SPARK_WORKER_INSTANCES=5
EXECUTOR_MEMORY=2G

# Train configuration
EPOCHS=2

# Input and output and not "hdfs://" pre-ffix
INPUT_DATA=/user-profile/yongxi/spark/input/mnist/csv/train
MODEL_DIR=/tmp/yongxi/tfoutput/mnist_model
EXPORT_DIR=/tmp/yongxi/tfoutput/mnist_export

# MODEL_DIR=/user-profile/yongxi/spark/tfoutput/mnist_model
# EXPORT_DIR=/user-profile/yongxi/spark/tfoutput/mnist_export

# INPUT_DATA=hdfs://opera/user-profile/yongxi/spark/input/mnist/csv/train
# MODEL_DIR=hdfs://opera/user-profile/yongxi/spark/tfoutput/mnist_model
# EXPORT_DIR=hdfs://opera/user-profile/yongxi/spark/tfoutput/mnist_export

# For TensorFlow 2.x (git checkout master)
# if MODLE_DIR exist then remove else skip
# if $(hadoop fs -test -d ${MODEL_DIR}); 
#     then sudo -u hdfs hadoop fs -rm -r -skipTrash ${MODEL_DIR}; echo "already remove the directory."
# else 
#     echo "there is no directory named ${MODEL_DIR}"; 
# fi

# if $(hadoop fs -test -d ${EXPORT_DIR}); 
#     then sudo -u hdfs hadoop fs -rm -r -skipTrash ${EXPORT_DIR}; echo "already remove the directory."
# else 
#     echo "there is no directory named ${EXPORT_DIR}"; 
# fi

sudo -u profile hadoop fs -rm -r -f -skipTrash ${MODEL_DIR}/*
sudo -u profile hadoop fs -rm -r -f -skipTrash ${EXPORT_DIR}/*


sudo -u profile ${SPARK_HOME}/bin/spark-submit \
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
                    --archives "../${CONDAENV}.zip#${CONDAENV}_zip" \
                    --jars ${TFCONNECTOR},${TFHADOOP} \
                    ./examples/mnist/keras/mnist_spark.py \
                        --cluster_size ${SPARK_WORKER_INSTANCES} \
                        --epochs ${EPOCHS} \
                        --images_labels ${INPUT_DATA} \
                        --model_dir ${MODEL_DIR} \
                        --export_dir ${EXPORT_DIR}
                    # ./try_spark.py \
