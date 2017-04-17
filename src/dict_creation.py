#
# Usage:
#   Stand-alone application: $SPARK_HOME_PATH/bin/spark-submit process.py
#   Interactive PySpark Shell: cat process.py | pyspark
#

from pyspark import SparkContext, SparkConf
import pickle

# Set up main entry point for Spark
conf = (SparkConf()
        .setMaster("local")
        .setAppName("My app")
        .set("spark.executor.memory", "1g"))
sc = SparkContext(conf=conf)
sc.setLogLevel("OFF")

print("\n\n\n")

# data_file = "file:///{path}/*.txt".format(path=directory)
data_file = "nips.vocab.trunc.txt"
rdd_data = sc.textFile(data_file).cache()
rdd_data = sc.parallelize(['DUMMY']) + rdd_data
rdd_data = rdd_data.zipWithIndex()
rdd_invert = rdd_data.map(lambda tok: tuple([tok[1], tok[0]]))

with open('mapping.pickle', 'wb') as handle:
    pickle.dump(rdd_invert.collectAsMap(), handle, protocol=pickle.HIGHEST_PROTOCOL)

sc.stop()
