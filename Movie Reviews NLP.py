#!/usr/bin/env python
# coding: utf-8

# # Loading & Pre-processing
# The required libraries are imported. The dataset is loaded and pre-processed, the syntax for training and test split for building ML models on, has been commented out for now.

# In[71]:


### import necessary packages
import pandas as pd
import math
import seaborn as sns
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# In[72]:


df = pd.read_json('xaa.json', lines=True) #DF made from JSON file 
##df.dropna #drop the NaN values if present
##Split dataset into training 80% and testing 20%, to be used later
### df_train, df_test = train_test_split(df, test_size=0.80, random_state=42)
df


# # Cleaning the data
# ### Dropping the columns with useless data

# In[73]:


## dropping the useless columns
df = df.drop(columns=['reviewerID', 'reviewTime', 'reviewerName'])


# In[74]:


df.describe()  ### Describe


# ## Observation one: a scatter 
# Plotting the helpfulness of reviews over the range of inputs:

# In[75]:


help_mat = df.loc[:, 'helpful'] 
 ## Y-plot, range


# In[76]:


## take random 100 values and plot probability

ran_help = help_mat.sample(n=300)
len(ran_help)


# In[77]:


prob = []
for i in range(len(ran_help)):
    key = list(ran_help.keys())[i]
    if(ran_help[key][1]!=0):
        prob.append(ran_help[key][0]/ran_help[key][1]) ## X-plot, the probabilities of a review being helpful
y=list(range(0,len(prob)))
plt.figure(figsize=(20,10))
plt.plot(y, prob)
plt.xlabel('Helpfulness Probability Trend', fontsize=18)


# # Observation two:
# Taking the average of helpful reviews for each product

# In[78]:


v = df.groupby('asin')['asin'].count() ## 44560

#x = pd.DataFrame(v.tolist(),columns=['count'])
#len(x[x['count']>100])
v = pd.DataFrame(v.tolist(),columns=['count'])
r = v[:100]
r.sum(axis=0)   


# In[79]:


df[:4123]


# In[80]:


df.loc[4123,'asin']   ##----> Proves that there are whole groups of products TILL 4122


# In[95]:


tot_rate=0
g_rate = 0
b_rate = 0
count=0
count_list = []
avg=0.0
asin=[]
prodID_list=[]
avgRate_list=[]
good_rate_count_list=[]
bad_rate_count_list=[]
rate_per_product = [] ## Percentage of good rating per product
good_rate_count=0
bad_rate_count=0
for i in range(4123):  ##or take the entire set
    if(df.iloc[i,0] == df.iloc[i+1,0]):## repeat for the same asin
        asin = str(df.iloc[i,0])
        if(df.iloc[i,1][0]>=df.iloc[i,1][1]): ## in case more helpful than unhelpful rating received for the review
            tot_rate = tot_rate+df.iloc[i,3]
            count=count+1
        if(df.iloc[i,3]>=4):
            g_rate = g_rate+df.iloc[i,3]
            good_rate_count=good_rate_count+1
        else:
            b_rate = b_rate+df.iloc[i,3]
            bad_rate_count=bad_rate_count+1
            
            
    else:
        asin = str(df.iloc[i+1,0])
        if(df.iloc[i,1][0]>=df.iloc[i,1][1]): ## in case more helpful than unhelpful rating received for the review
            tot_rate = tot_rate+df.iloc[i,3]
            count=count+1
        if(df.iloc[i,3]>=4):
            g_rate = g_rate+df.iloc[i,3]
            good_rate_count=good_rate_count+1
            
        else:
            b_rate = b_rate+df.iloc[i,3]
            bad_rate_count=bad_rate_count+1
            
            
        if(count!=0):
            avg = tot_rate/count
            #print("The average helpful rating for the product ID ", asin, "is ",avg)
        good_rate_count_list.append(good_rate_count)
        bad_rate_count_list.append(bad_rate_count)
        rate_per_product.append((good_rate_count)/(good_rate_count+bad_rate_count) * 100) ## weighted average
        prodID_list.append(asin)
        count_list.append(count)
        avgRate_list.append(round(avg,2))
        ### initialise again
        tot_rate=0
        count=0
        g_rate = 0
        b_rate = 0
        good_rate_count=0
        bad_rate_count=0
        avg=0.0



df_list = [prodID_list, avgRate_list]
df_list = np.swapaxes(df_list, 0, 1).tolist()
helpful_rate_df = DataFrame(df_list, columns= ['ProductID', 'Average helpful good rating'])



# ### Percentage Good Rating per product

# In[96]:


prods = [x for x in range(len(prodID_list))]
rpp_per_product_df = pd.DataFrame({"Product": prodID_list, "%age >=4 Rating": rate_per_product})
rpp_per_product_df


# ### Grouped bar graph to visualise the results over the set of first 100 products

# In[99]:



x1 = good_rate_count_list
x2 = bad_rate_count_list

#plt.hist([x1, x2], bins = 100,color = color_list, label=rate_type)


kwargs = dict(hist_kws={'alpha':.01}, kde_kws={'linewidth':5})
bar = [x for x in range(100)]

plt.bar(bar,x1, 0.25, alpha=1.0, color='green')


    # Create a bar with post_score data,
plt.bar(bar,x2, 0.25, alpha=0.9, color='red')
plt.xticks(rotation=60)
plt.legend()


# ### Grouped Bar graph: smaller scale

# In[60]:


### grouped bar graph
rate_type = ["Good", "Bad"]
color_list = ['Blue', 'Red']
x1 = good_rate_count_list
x2 = bad_rate_count_list

#plt.hist([x1, x2], bins = 100,color = color_list, label=rate_type)


kwargs = dict(hist_kws={'alpha':.6}, kde_kws={'linewidth':2})
bar = ["Prod1", "Prod2", "Prod3", "Prod4", "Prod5", "Prod6", "Prod7", "Prod8", "Prod9", "Prod10"]
plt.bar(bar,x1[:10], 0.25, alpha=0.4, color='green')


    # Create a bar with post_score data,
plt.bar(bar,x2[:10], 0.25, alpha=0.3, color='red')
plt.xticks(rotation=60)
plt.legend()


# # Observation three:
# Comparing the number of good raters (above 4) for each individual product in the form of a **bar graph**

# In[61]:


### Name the products
prodName_list=[]
name=""
for i in range(len(prodID_list)):
    name="Prod"+str(i+1)
    prodName_list.append(name)


# In[62]:


### Plot bar graph

fig = plt.figure()
ax = fig.add_axes([0,0,1,1]) 
x = prodName_list
y = good_rate_count_list
ax.bar(x[:20],y[:20])
plt.xticks(rotation=45)
plt.show()


# ## Cluster analysis

# In[63]:


from sklearn.cluster import KMeans


# In[64]:


data = {"Length":len_list, "Rating":overall_list}
df_temp = pd.DataFrame(data)
X = df_temp.sample(n=50).iloc[:,[0,1]]


# In[ ]:


df_temp


# In[ ]:


Kmean = KMeans(n_clusters=5)
Kmean.fit(X)
Kmean.cluster_centers_


# In[ ]:


plt.scatter(X.iloc[:,0], X.iloc[:,1], c=Kmean.labels_, cmap='rainbow')


# # Relation between length and rating: an analysis 

# ### Probability for a long review to have a good rating

# In[101]:


import statistics
sample_df = df.sample(n=1000)
rev_list = sample_df['reviewText']
helpful_count=0
long_rev = []
good_count=0
bad_count=0
good_list=[]
bad_list=[]
prob_long_good=[]
good_helpful = 0
good_unhelpful = 0
bad_helpful = 0
bad_unhelpful = 0
l = [len(x) for x in rev_list]
m = statistics.median(l)
for i in range(len(rev_list)):
    key = list(rev_list.keys())[i]
    if((len(rev_list[key]))>=m):##consider mean/mode
        
        long_rev.append(rev_list[key])
        if(sample_df.loc[key, 'overall']>=4):
            good_list.append(sample_df.loc[key,'overall'])
        else:
            bad_list.append(sample_df.loc[key,'overall'])
            
## probability of a good review being marked as helpful 
for i in range(len(rev_list)):
    key = list(rev_list.keys())[i]
    if(sample_df.loc[key,'overall']>=4): ## if good-rated 
        if(sample_df.loc[key, 'helpful'][0]>(sample_df.loc[key, 'helpful'][1]-sample_df.loc[key, 'helpful'][0])): ## if helpful
            good_helpful = good_helpful + 1
        else:
            good_unhelpful = good_unhelpful + 1
    else:
        if(sample_df.loc[key, 'helpful'][0]>(sample_df.loc[key, 'helpful'][1]-sample_df.loc[key, 'helpful'][0])): ## if helpful
            bad_helpful = bad_helpful + 1
        else:
            bad_unhelpful = bad_unhelpful + 1


# In[103]:


p_good_helpful = good_helpful/(good_helpful+good_unhelpful+bad_helpful+bad_unhelpful)
p_good_unhelpful = good_unhelpful/(good_helpful+good_unhelpful+bad_helpful+bad_unhelpful)
p_bad_helpful = bad_helpful/(good_helpful+good_unhelpful+bad_helpful+bad_unhelpful)
p_bad_unhelpful = bad_unhelpful/(good_helpful+good_unhelpful+bad_helpful+bad_unhelpful)


# In[107]:


p_good_unhelpful


# **BOTH GOOD AND BAD RATED REVIEWS HAVE MORE UNHELPFUL RATE THAN HELPFUL --> PEOPLE HAVE A TENDENCY TO DOWN-RATE IRRESPECTIVE OF THE CONTENT OF THE REVIEW**

# In[108]:


prob = len(good_list)/(len(bad_list)+len(good_list)) 
prob


# In[69]:


helpful = 0
unhelpful = 0
for i in range(len(rev_list)):
    key = list(rev_list.keys())[i]
    if((len(rev_list[key]))>m):##consider mean/mode
        
        long_rev.append(rev_list[key])
        if(sample_df.loc[key, 'helpful'][0]>(sample_df.loc[key, 'helpful'][1]-sample_df.loc[key, 'helpful'][0])):
                helpful = helpful+1
        else:
                unhelpful = unhelpful + 1
        


# In[70]:


prob = helpful/(helpful+unhelpful)
prob


# * ~80% of the longer reviews have OVERALL rating as GOOD (>=4)
# * Longer reviews are MORE LIKELY to be marked as HELPFUL over SHORTER REVIEWS
# * Most people have UP-VOTED longer reviews and DOWN-VOTED shorter reviews
# 
# 

# In[71]:


corr_df = df.sample(n = 1000)
corr_df


# In[72]:


##
corr_rev_list = corr_df.iloc[:,2]
rate_list = corr_df.iloc[:,3]
help_list = corr_df['helpful']
overall_list = corr_df['overall']

len_list = []
for i in corr_rev_list:
    len_list.append(len(i))
l = [x for x in range(1000)]

x = overall_list
y = len_list
ser_x = pd.Series(x, index=l)
ser_y = pd.Series(y, index=l)
ser_x.corr(ser_y, method='kendall') ## Kendall's tau correlation coeff


# There is a **low negative** correlation between the length and the rating. DISSATISFIED CUSTOMERS SEEM TO WRITE LONGER REVIEWS THAN SATISFIED ONES.

# ### Line graph between length and overall

# In[73]:


##from scipy.spatial import distance_matrix
##dist_df = pd.DataFrame({'Length': len_list, 'Overall': overall_list})
##pd.DataFrame(distance_matrix(dist_df.values, dist_df.values), index=dist_df.index, columns=dist_df.index)


# In[74]:


plt.plot(overall_list[:100],len_list[:100])
plt.show()


# There is **no linear** trend in the relation of length and review. Some trends:
# 
# *  5 rates go over the entire range of review length (nearly)
# *  4.5 rates do not go beyond 5000 characters of rev length
# *  3 star rates barely cross a length of 2500 characters
# 
# 

# ### Scatter plot between Length and Rating

# In[75]:


plt.scatter(y,x)
plt.xlabel("Length of review", fontsize=14)
plt.ylabel("Rating", fontsize=14)
plt.show()


# * *Continued later in NLP Section*

# # NLP: Pre-processing the text

# In[76]:


## remove non-essential characters of review text
df['reviewText'] = df['reviewText'].str.replace("[^a-zA-Z#\ '/()[]]", " ")

## convert all the words to lower case entirely
df['reviewText'] = df['reviewText'].str.lower()       


# In[77]:


## take a random DF of 1000 samplets from original df
r_df = df.sample(n=1000)


# ### Plot frequent words occurring in the set of reviews

# In[78]:


nltk.download('stopwords')
from nltk.corpus import stopwords
sw = stopwords.words('english')
from nltk import FreqDist


# In[122]:


# function to plot most frequent terms

def freq_words(x, terms = 30):
    all_words = ' '.join([text for text in x])
    all_words = all_words.split()

    fdist = FreqDist(all_words)
    words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})

  # selecting top 20 most frequent words
    d = words_df.nlargest(columns="count", n = terms) 
    plt.figure(figsize=(20,5))
    ax = sns.barplot(data=d, x= "word", y = "count")
    ax.set(ylabel = 'Count')
    plt.show()


# In[123]:


## removes words lesser than 3 digits
r_df['reviewText'] = r_df['reviewText'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))


# In[124]:


freq_words(r_df['reviewText']) ### displays top 20 most frequent words


# In[82]:


## function to remove stop words
def remove_stopwords(rev):
    rev_new = " ".join([i for i in rev if i not in sw])
    return rev_new


# In[83]:


reviews = [remove_stopwords(r.split()) for r in r_df['reviewText']]


# In[84]:


freq_words(reviews,35) ### after removing stopwords


# Words like **movie** and **film** are at the top, meaning they occur most often during the review. Words with the highest frequency gives us some idea about the **context** of the review.

# ## Next step: Text Summary

#          

# In[109]:


from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize, sent_tokenize


# In[114]:


# importing libraries 
import statistics
import nltk 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize, sent_tokenize 
   
# Input text - to summarize  
def summarize(text):
    # Tokenizing the text 
    stopWords = set(stopwords.words("english")) 
    words = word_tokenize(text) 
   
# Creating a frequency table to keep the  
# score of each word 
   
    freqTable = dict() 
    for word in words: 
        word = word.lower() 
        if word in stopWords: 
            continue
        if word in freqTable: 
            freqTable[word] += 1
        else: 
            freqTable[word] = 1
   
## Creating a dictionary to keep the score 
# of each sentence 
    sentences = sent_tokenize(text) 
    sentenceValue = dict() 
   
    for sentence in sentences: 
        for word, freq in freqTable.items(): 
            if word in sentence.lower(): 
                if sentence in sentenceValue: 
                    sentenceValue[sentence] += freq 
                else: 
                    sentenceValue[sentence] = freq 
   
   
   
    sumValues = 0
    for sentence in sentenceValue: 
         sumValues += sentenceValue[sentence] 
   
# Average value of a sentence from the original text 
   
    average = int(sumValues / len(sentenceValue)) 
   
# Storing sentences into our summary. 
    summary = "" 
    for sentence in sentences: 
        if (sentence in sentenceValue) and (sentenceValue[sentence] > (1.2 * average)):
             summary += sentence
    return summary


# In[115]:


summarize("There are many techniques available to generate extractive summarization to keep it simple, I will be using an unsupervised learning approach to find the sentences similarity and rank them. Summarization can be defined as a task of producing a concise and fluent summary while preserving key information and overall meaning. One benefit of this will be, you don’t need to train and build a model prior start using it for your project. It’s good to understand Cosine similarity to make the best use of the code you are going to see. Cosine similarity is a measure of similarity between two non-zero vectors of an inner product space that measures the cosine of the angle between them. Its measures cosine of the angle between vectors. The angle will be 0 if sentences are similar.")


# ## Summarizing long reviews and checking percentage of reduction

# 
# 
# 
# 

# In[125]:


df['reviewText'] = df['reviewText'].str.replace("[^a-zA-Z#\ '/()[]]", " ") ##TBD : remove '/' separately
rev_list = df.sample(n=100)['reviewText']


# In[126]:



l = [len(x) for x in rev_list]
m = statistics.median(l) ## median length
long_rev = []
count_or = 0
for i in rev_list:
    if(len(i)>m): 
        long_rev.append(i)
        count_or = count_or + len(i)


# In[127]:


summary_list = []
count = 0
for x in long_rev:
    summary_list.append(summarize(x))
    
for i in range(len(long_rev)):
    count_or = count_or+len(long_rev[i])
    count = count + len(summarize(long_rev[i]))


# In[128]:


percentage_reduction = ((count_or - count)/count_or)*100
percentage_reduction


# In[131]:


long_rev[10]


# In[132]:


summary_list[10]


# 
# 
# 
# 

#  
#  
#   

#       

# # NLP: NLTK,  Sentiment Intensity Analyzer
# 
# Sentimental analysis to determine the correlation with length of the review. The result we had derived without using NLP earlier has only been clarified using NLP. We have followed the steps:
# 
# 1. Obtained random samplets of 1000 data points each for length and their corresponding reviews
# 2. Obtained the probabilities for the reviews to be positive/negative/neutral/combined
# 3. Correlated the positive probabilties to the lengths of the review and found a MEDIUM NEGATIVE value, confirming the results we had earlier derived from the numerical rating and length correlation: Users are likely to write LESS POSITIVELY when they write longer reviews
# 4. Correlated the negative probabilities to the length of the text and found a HIGH POSITIVE correlation constant, further establishing the statement above.

# In[113]:


import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


corr_rev_list = df.sample(n=1000)['reviewText']
len_list = []
for i in corr_rev_list:
    len_list.append(len(i))
sid = SentimentIntensityAnalyzer()
sent = corr_rev_list.apply(lambda x: sid.polarity_scores(x))


# In[114]:


sent


# In[115]:


pos_value = 0
prob_pos=[]
for keys in range(0,1000):
    key = list(sent.keys())[keys]
    prob_pos.append(sent[key]['pos'])


# In[116]:


x = prob_pos
y = len_list
r = np.corrcoef(x, y)
r


# In[117]:


plt.scatter(x,y,color="green")
plt.xlabel("Positive Sentiment Probability", fontsize=18)
plt.ylabel("Length of the review", fontsize=18)


# In[118]:


neg_value = 0
neg_pos=[]
for keys in range(0,1000):
    key = list(sent.keys())[keys]
    neg_pos.append(sent[key]['neg'])


# In[119]:


x = neg_pos
y = len_list
r = np.corrcoef(x, y)
r


# In[120]:


plt.scatter(x,y, color="red")
plt.xlabel("Negative Sentiment Probability", fontsize=18)
plt.ylabel("Length of the review", fontsize=18)


# In[121]:


### Compare both the above charts 

plt.figure(figsize=(10,3))
#ax = fig.add_subplot(11)
plt.scatter(prob_pos, y, marker='o',
color='green', alpha=0.7, label='Positive Probability')
plt.scatter(neg_pos, y, marker='o',
color='red', alpha=0.3, label='Negative Probability')
plt.xlabel('Positive and Negative Sentiment Probabilies')
plt.ylabel('Length of review')
plt.legend(loc='upper right')
plt.show()


# Up to a probability of approximately 0.15, both +ve and -ve sentiment probabilities behave kind of equally wrt the length of the review. Beyond that, positive sentiments take over and prevail upto probability of 0.5 for ~6000-7000 max, excepting the outliers which are seen to reach up to 20,000 in length (with positive sentimental probability of 0.2 and negative sentimental probability of ~0.05)

# # Context Analysis and Text Generation: an idea 
# 
# ## Aim: Discourse analysis on speech
# Step 1: Context analysis
# Step 2: To check the probability of a particular word appearing in a particular context.
# 

# In[95]:


from IPython.display import Image
Image(filename='context.png')


# # The Final Step: Auto-response generation to reviews

# In[96]:


from IPython.display import Image
Image(filename='reply.png')


# The above image shows a reply from the creator in the Google Play Store on a particular app. Our aim is to simulate the reply from the original creator and to **auto-generate a reply** to the review by the user, using an inbuilt sentimental analyzer for the review.
# 
# The alternative: 
# The system can be simulated by a simple rule-based system which displays automated text depending on the percentage of sentimental quantities, i.e., based on pre-defined rules.
# 
# The drawback: 
# The replies will NOT be personalized, in other words, they won't imitate a human's reaction to the review.
# 
# Solution: **A customized LTSM Model**
# 
# 
# AN ATTEMPT AT CONTEXT RECOGNITION: **CBOW**

# # The Proposed Model
# 
# **LTSM using Keras and Tensorflow**: This will create the actual text generator and/or predictive text generator.
# 
# **The training dataset**: A training dataset on a set of possible replies AND the following, in addition:
# 
# 1. Analysis from helpfulness and rating of the review in concern
# 2. Sentimental analysis results
# 3. COBW context analysis from the CONTRASTING text from the review's content
# 
# 

# In[97]:


Image(filename='rough.png')


# In[98]:


Image(filename='ans.png')


# **Problems:**
# 
# Too many typos, inaccurate analysis of text

# ## W2V

# In[99]:


from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
warnings.filterwarnings(action = 'ignore')


# In[100]:


import gensim
from gensim.models import Word2Vec


# In[101]:


st=""
for keys in range(0,1000):
    key = list(sent.keys())[keys]
    st=st+corr_rev_list[key]


# In[102]:


st


# In[103]:


import re
sample = st ### Random review from the dataset

### cleaning
pattern = ")(][`/,.!-:!*"
for char in pattern:
    sample = sample.replace(char,"")
sample = sample.replace("\'","")
sample = sample.replace("\"","")


sample


# In[112]:


data = []
for i in sent_tokenize(sample):
    temp=[]
    
    for j in word_tokenize(i):
        temp.append(j.lower())
    data.append(temp)


# In[113]:


model = gensim.models.Word2Vec(data, min_count=1,size=500, window=len(data), iter=10)
model.wv.vocab


# In[100]:


#model1.wv.similarity('the', 'scenery')

print(model)


# In[114]:


model.wv.similarity('bad', 'worst')


# In[138]:


model.wv.most_similar(positive='care')


# In[97]:


model['good']


# In[92]:


from sklearn.decomposition import PCA
X = model[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
plt.scatter(result[:, 0], result[:, 1])


# In[108]:


from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
# define training data
sentences = [['the', 'first', 'statement', 'is', 'the', 'one', 'here'],
			['this', 'is', 'the', 'second', 'sentence'],
			['and', 'another', 'sentence'],
			['we', 'have', 'one', 'more', 'statement', 'right', 'here'],
			['and', 'the', 'is', 'the','final', 'sentence']]
# train model
model = Word2Vec(sentences, min_count=1)
# fit a 2d PCA model to the vectors
X = model[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
# create a scatter plot of the projection
pyplot.scatter(result[:, 0], result[:, 1])
words = list(model.wv.vocab)
for i, word in enumerate(words):
	pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.title("Annotation of the words", fontsize=18)
pyplot.show()


# In[108]:


len(summary) ### 54% reduction in size


# In[106]:


len(sample)

