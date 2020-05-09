#!/usr/bin/bash
export SPARK_HOME=/Users/yongxiyang/Desktop/spark-2.4.4-bin-hadoop2.7
# export SPARK_HOME=/home/sdev/yongxi/env/spark-2.4.4-bin-hadoop2.7
${SPARK_HOME}/sbin/stop-slave.sh; ${SPARK_HOME}/sbin/stop-master.sh