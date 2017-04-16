#
# Usage:
#   Stand-alone application: $SPARK_HOME_PATH/bin/spark-submit process.py
#   Interactive PySpark Shell: cat process.py | pyspark
#

from pyspark import SparkContext, SparkConf
from bs4 import BeautifulSoup
from creole import creole2html


class Article(object):

    def __init__(self, title, text, links=None):
        self.title = title
        self.text = text
        self.links = links


def remove_html(article):
    soup = BeautifulSoup(article.text, "lxml")
    article.text = soup.get_text()
    return article


def parse_xml(line):
    soup = BeautifulSoup(line, "lxml")
    return Article(soup.page.title, soup.page.text)


def remove_creole(article):
    article.text = creole2html(article.text)
    return article

# Set up main entry point for Spark
conf = (SparkConf()
        .setMaster("local")
        .setAppName("My app")
        .set("spark.executor.memory", "1g"))
sc = SparkContext(conf=conf)
sc.setLogLevel("OFF")

print("\n\n\n")
# directory = input("Data dump directory absolute path: ")
directory = "/home/madhav/Downloads/DBMS"

data_file = "file:///{path}/*.txt".format(path=directory)
rdd_data = sc.textFile(data_file).cache()
rdd_data_string = [rdd_data.reduce(lambda a, b: a + '\n' + b)]

rdd_xml = sc.parallelize(rdd_data_string, 16)\
        .flatMap(lambda line: line.split('</page>'))\
        .map(lambda line: line + '</page>')\
        .map(parse_xml)\
        .map(remove_creole)\
        .map(remove_html)

# Pre-processing and parsing out creole text
rdd_counter = rdd_xml.map(creole2html).map(remove_html)

counts = rdd_counter.flatMap(lambda line: line.split(" ")) \
    .map(lambda word: (word, 1)) \
    .reduceByKey(lambda a, b: a + b)
print(counts.collect())
# counts.saveAsTextFile("word-counts.txt")
sc.stop()
