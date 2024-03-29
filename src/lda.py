import os

from pyspark.sql import SQLContext, Row
from pyspark.ml.feature import CountVectorizer
from pyspark.mllib.clustering import LDA, LDAModel
from pyspark import SparkContext, SparkConf
from pyspark.mllib.linalg import Vectors

conf = (SparkConf()
        .setMaster("local")
        .setAppName("My")
        .set("spark.executor.memory", "1g"))
sc = SparkContext(conf=conf)
sc.setLogLevel("OFF")

sqlContext = SQLContext(sc)
path = 'clean_test.txt'  # path of the txt file

data = sc.textFile(path).zipWithIndex().map(lambda line: Row(idd=line[1], words=line[0].split(" ")))
os.system('rm -f metastore_db/dbex.lck')
docDF = sqlContext.createDataFrame(data)

Vector = CountVectorizer(inputCol="words", outputCol="vectors")
Vector = Vectors.fromML(Vector)
model = Vector.fit(docDF)
result = model.transform(docDF)

corpus_size = result.count()  # total number of words
corpus = result.select("idd", "vectors").rdd.map(lambda line: [line[0], line[1]]).cache()

# Cluster the documents into three topics using LDA
ldaModel = LDA.train(corpus, k=3, maxIterations=100, optimizer='online')
topics = ldaModel.topicsMatrix()
vocabArray = model.vocabulary

wordNumbers = 10  # number of words per topic
topicIndices = sc.parallelize(ldaModel.describeTopics(maxTermsPerTopic=wordNumbers))


def topic_render(topic):  # specify vector id of words to actual words
    terms = topic[0]
    result = []
    for i in range(wordNumbers):
        term = vocabArray[terms[i]]
        result.append(term)
    return result


topics_final = topicIndices.map(lambda topic: topic_render(topic)).collect()

for topic in range(len(topics_final)):
    print("Topic" + str(topic))
    for term in topics_final[topic]:
        print(term)
    print('\n')
