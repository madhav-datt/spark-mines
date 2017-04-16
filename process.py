#
# Usage:
#   Stand-alone application: $SPARK_HOME_PATH/bin/spark-submit process.py
#   Interactive PySpark Shell: cat process.py | pyspark
#

from pyspark import SparkContext, SparkConf
from bs4 import BeautifulSoup
from creole import creole2html
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import re

stop = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')


class Article(object):
    """
    Class with wikipedia article instance
    """
    def __init__(self, title, text, links=None):
        self.title = title
        self.text = text
        self.links = links


def remove_html(article):
    title, text = article
    soup = BeautifulSoup(text, "lxml")
    text = soup.get_text()
    return title, text


def parse_xml(line):
    soup = BeautifulSoup(line, "lxml")
    # return soup.page.title.string
    # return tuple([soup.page.title, soup.page.text])
    try:
        title = soup.page.title.string
    except AttributeError:
        title = ""
    try:
        text = soup.page.text
    except AttributeError:
        text = ""
    return title, text


def remove_creole(article):
    title, text = article
    text = creole2html(text)
    return title, text


def pre_process(token):
    """
    Remove punctuation, numbers and stopwords from text tokens
    :param token: token to be processed
    :return: cleaned token string
    """

    token = ' ' + token.lower() + ' '
    tmp_token = re.sub("[^a-zA-Z\s]", ' ', token)
    for word in tmp_token.split():
        replace_word = ' ' + str(word) + ' '
        if word in stop:
            tmp_token = tmp_token.replace(replace_word, ' ')
        tmp_token = tmp_token.replace(replace_word, ' ' + stemmer.stem(word) + ' ')

    return ' '.join(tmp_token.split())

# Set up main entry point for Spark
conf = (SparkConf()
        .setMaster("local")
        .setAppName("My app")
        .set("spark.executor.memory", "1g"))
sc = SparkContext(conf=conf)
sc.setLogLevel("OFF")

print("\n\n\n")
# directory = input("Data dump directory absolute path: ")
# directory = "/home/madhav/Downloads/DBMS"

# data_file = "file:///{path}/*.txt".format(path=directory)
data_file = 'test.txt'
rdd_data = sc.textFile(data_file).cache()
rdd_data_string = [rdd_data.reduce(lambda a, b: a + '\n' + b)]

rdd_xml = sc.parallelize(rdd_data_string, 16)\
        .flatMap(lambda line: line.split('</page>'))\
        .map(lambda line: line + '</page>')\
        .map(parse_xml)\
        .map(remove_creole)\
        .map(remove_html)

# Pre-processing and parsing out creole text
# rdd_counter = rdd_xml.map(creole2html).map(remove_html)
rdd_counter = rdd_xml.map(lambda article: article[1])\
    .map(pre_process)

counts = rdd_counter.flatMap(lambda line: line.split(" ")) \
    .map(lambda word: (word, 1)) \
    .reduceByKey(lambda a, b: a + b)

print(counts.collect())
# counts.saveAsTextFile("word-counts.txt")
sc.stop()
