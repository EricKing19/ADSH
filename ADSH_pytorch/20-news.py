from sklearn.datasets import fetch_20newsgroups_vectorized
from sklearn.datasets import fetch_20newsgroups
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.io



train_dataset = fetch_20newsgroups(subset = 'train')
test_dataset = fetch_20newsgroups(subset = 'test')

#x_train, x_test, y_train, y_test = train_test_split (news.data,news.target,test_size = 0.25,random_state = 33)

tfid_vec = TfidfVectorizer(sublinear_tf = True, max_df = 0.5,stop_words = 'english',max_features=50000)
x_tfid_train = tfid_vec.fit_transform(train_dataset.data)
x_tfid_test = tfid_vec.transform(test_dataset.data)

scipy.io.savemat('data_news_20_50000.mat',{'x_test':(x_tfid_test).T,'y_test':test_dataset.target,'x_train':(x_tfid_train).T,'y_train':train_dataset.target})


tfidf_train_3 = fetch_20newsgroups_vectorized(subset = 'train');
tfidf_test_3 = fetch_20newsgroups_vectorized(subset = 'test');


print "the shape of train is "+repr(tfidf_train_3.data.shape)
print "the shape of test is "+repr(tfidf_test_3.data.shape)

dataset = fetch_20newsgroups(subset = 'train')
dataset = fetch_20newsgroups(subset = 'test')

vectorizer = TfidfVectorizer(max_features = 2000)
vectors = vectorizer.fit_transform(dataset.data)


scipy.io.savemat('data_news_20.mat',{'x_test':(tfidf_test_3.data).T,'y_test':tfidf_test_3.target,'x_train':(tfidf_train_3.data).T,'y_train':tfidf_train_3.target})


