"""
Ethan McIlveen, March 2023

Loads an example analysis documents and labels, used to identify sincerity(i.e. thank you), greetings (i.e. hello), and farewells (i.e. goodbye). This also has ~100 other lines of texts that are random questions/statements, used to determine if we should continue down the pipeline

# 0 = opening statement i.e. hello
# 1 = closing statement i.e. goodbye
# 2 = thankful/grateful statement
# 3 = question/statement
"""
### Load docs and labels
filename = "example_corpus.txt"
docs = []
labels = []
with open(filename) as file:
    for line in file:
        line = line.strip()
        labels.append(int(line[-1]))
        docs.append(line[:-2].strip())

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

split = train_test_split(docs, labels)
train_docs, test_docs, train_labels, test_labels = split

## vectorize
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
train_data = vectorizer.fit_transform(train_docs)
test_data = vectorizer.transform(test_docs)


## report more metrics
print(train_data.shape[1],"features")
print(train_data.shape[0],"training examples")
print(test_data.shape[0],"testing examples")
print()


## create and train the classifier
clf = MLPClassifier(max_iter = 1000)
clf.fit(train_data, train_labels)


## get predictions for unseen examples
pred = clf.predict(test_data)

##export to .joblib files
from joblib import dump
dump(clf, 'example_corpus.joblib')
dump(vectorizer, 'vectorizer.joblib')
