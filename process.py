#
# Usage:
#   Stand-alone application: $SPARK_HOME_PATH/bin/spark-submit process.py
#   Interactive PySpark Shell: cat process.py | pyspark
#

from pyspark import SparkContext, SparkConf
from bs4 import BeautifulSoup
from creole import creole2html


def remove_html(html_line):
    soup = BeautifulSoup(html_line)
    return soup.title.string, soup.get_text()


# Set up main entry point for Spark
conf = (SparkConf()
        .setMaster("local")
        .setAppName("My app")
        .set("spark.executor.memory", "1g"))
sc = SparkContext(conf=conf)
sc.setLogLevel("OFF")

directory = input("Data dump directory absolute path: ")

data_file = "file:///{path}/*.txt".format(path=directory)
rdd_data = sc.textFile(data_file).cache()
rdd_data_string = [sc.reduce(lambda a, b: a + '\n' + b)]
rdd_xml = sc.parallelize(rdd_data_string, 16).flatMap(lambda line: line.split('</page>'))\
    .map(lambda line: line + '</page>')


# Pre-processing and parsing out creole text
rdd_data = rdd_data.map(creole2html).map(remove_html)

counts = rdd_data.flatMap(lambda line: line.split(" ")) \
    .map(lambda word: (word, 1)) \
    .reduceByKey(lambda a, b: a + b)
counts.saveAsTextFile("word-counts.txt")
sc.stop()
