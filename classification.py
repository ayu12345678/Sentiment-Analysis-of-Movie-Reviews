import pandas as pd
import matplotlib.pyplot as plt
import nltk
import numpy as np
nltk.download('punkt')


data=pd.read_csv(r"C:\Users\Ayushi\Downloads\trainer.csv")
test=pd.read_csv(r"C:\Users\Ayushi\Downloads\tester.csv")
print(data.head())
print(data.info())
print(data.Sentiment.value_counts())
Sentiment_count=data.groupby('Sentiment').count()
plt.bar(Sentiment_count.index.values, Sentiment_count['Phrase'])
plt.xlabel('Review Sentiments')
plt.ylabel('Number of Review')

test_id = test['PhraseId']
num_missing_desc = data.isnull().sum()    # No. of values with msising descriptions
print('Number of missing values: ' + str(num_missing_desc))
df = data.dropna()

from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
#tokenizer to remove unwanted elements from out data like symbols and numbers
token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize)
text_counts= cv.fit_transform(df['Phrase'].values.astype('U'))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    text_counts, df['Sentiment'], test_size=0.3, random_state=1)


from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
# Model Generation Using Multinomial Naive Bayes
clf = MultinomialNB().fit(X_train, y_train)
predicted= clf.predict(X_test)
print("MultinomialNB Accuracy:",metrics.accuracy_score(y_test, predicted))

output = pd.DataFrame( data={"PhraseId":test["PhraseId"], "Sentiment":predicted} )
output.to_csv( "Submission", index=False, quoting=3 )
