#
# Usage:
#   Stand-alone application: $SPARK_HOME_PATH/bin/spark-submit process.py
#   Interactive PySpark Shell: cat process.py | pyspark
#

from pyspark import SparkContext, SparkConf
from bs4 import BeautifulSoup
from creole import creole2html
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import pickle
import io

stop = set(stopwords.words('english'))
lem = WordNetLemmatizer()


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
    line = re.sub('<sha1(.)*?>(.)*?</sha1>', ' ', line, count=0, flags=re.DOTALL)
    line = re.sub('<username(.)*?>(.)*?</username>', ' ', line, count=0, flags=re.DOTALL)
    line = re.sub('<id(.)*?>(.)*?</id>', ' ', line, count=0, flags=re.DOTALL)
    line = re.sub('<model(.)*?>(.)*?</model>', ' ', line, count=0, flags=re.DOTALL)
    line = re.sub('<timestamp(.)*?>(.)*?</timestamp>', ' ', line, count=0, flags=re.DOTALL)
    line = re.sub('<format(.)*?>(.)*?</format>', ' ', line, count=0, flags=re.DOTALL)
    line = re.sub('<comment(.)*?>(.)*?</comment>', ' ', line, count=0, flags=re.DOTALL)
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


def parse_links(article):
    pattern = re.compile("\[\[([^\]]*)\]\]")
    title, text = article
    links = pattern.findall(text)

    for i in range(len(links)):
        if '|' in links[i]:
            links[i] = links[i][:links[i].index('|')]
        if '#' in links[i]:
            links[i] = links[i][:links[i].index('#')]

    return title, links


def pre_process(token):
    """
    Remove punctuation, numbers and stopwords from text tokens
    :param token: token to be processed
    :return: cleaned token string
    """

    token = re.sub('[^\sa-zA-Z]', ' ', token)  # ''.join(e for e in token if e.isalnum())
    token = token.lower()
    return ' '.join([word for word in token.split() if word not in stop])


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
data_file = 'data2.txt'
rdd_data = sc.textFile(data_file).cache()
rdd_data_string = [rdd_data.reduce(lambda a, b: a + '\n' + b)]

rdd_xml = sc.parallelize(rdd_data_string, 16) \
    .flatMap(lambda line: line.split('</page>')) \
    .map(lambda line: line + '</page>') \
    .map(parse_xml)

# Parse links from text into separate dictionary
rdd_links = rdd_xml.map(parse_links)
links_dict = rdd_links.collectAsMap()
with io.open('pagerank.pickle', 'wb') as handle:
    pickle.dump(links_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

rdd_xml = rdd_xml.map(remove_creole) \
    .map(remove_html)

# Pre-processing and parsing out creole text
# rdd_counter = rdd_xml.map(creole2html).map(remove_html)
rdd_counter = rdd_xml.map(lambda article: article[1]) \
    .map(pre_process)

rdd_title = rdd_xml.map(lambda article: article[0])
text_list = rdd_counter.collect()
title_list = rdd_title.collect()

with io.open('clean_test.txt', 'w') as clean_test:
    for article in zip(title_list, text_list):
        final_text = "<DOC> <DOCNO> {title} </DOCNO> <TEXT> {text} </TEXT> </DOC>".format(title=article[0],
                                                                                          text=article[1])
        clean_test.write(final_text + "\n")

counts = rdd_counter.flatMap(lambda line: line.split(" ")) \
    .map(lambda word: (word, 1)) \
    .reduceByKey(lambda a, b: a + b)

print(counts.collect())
sc.stop()
