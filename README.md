# NLP-Review-Insights

The project aims at understanding and drawing insights from customer reviews about a certain product. The chosen genre to perform the analysis has been chosen to be Movies. 

## Data cleaning

Missing data or data with discrepancies dropped and replaced with managed values for ease of analysis.
Stop words are identified by removed using enhancements over the basic nltk stopwords library
Sentences in the review were cleaned of punctuations and other irrelevant symbols

## EDA Done based on

### Following visualisations:

##### Grouped Bar chart for visualization of the good and bad reviews comparison
##### Maximum good ratings received by which product


### Insights & Probability analysis:

##### Probability of a longer review to have good rating was found to be considerably high
##### Identified that people more often downrate ratings as unhelpful, in case of both good and bad reviews
##### ~80% of the longer reviews have OVERALL rating as GOOD (>=4)
##### Longer reviews are MORE LIKELY to be marked as HELPFUL over SHORTER REVIEWS
##### Most people have UP-VOTED longer reviews and DOWN-VOTED shorter reviews


## Word frequency analysis

##### Relevant words were found to be used the most frequenty, after removing stopwords

## NLTK and Text Summarization

##### Review summarization by excluding redundant and stop words, sentences and tokens for the ease of reading.
##### Sentiment intensity analyzer to understand the positivity/negativity of the review.
##### Findings after Sentimental analysis:
      Obtained random samplets of 1000 data points each for length and their corresponding reviews
      Obtained the probabilities for the reviews to be positive/negative/neutral/combined
      Correlated the positive probabilties to the lengths of the review and found a MEDIUM NEGATIVE value, confirming the results we had earlier derived from the         numerical rating and length correlation: Users are likely to write LESS POSITIVELY when they write longer reviews
      Correlated the negative probabilities to the length of the text and found a HIGH POSITIVE correlation constant, further establishing the statement above.


#### Conclusion drawn (89% correct)

##### Up to a probability of approximately 0.15, both +ve and -ve sentiment probabilities behave kind of equally wrt the length of the review. Beyond that, positive sentiments take over and prevail upto probability of 0.5 for ~6000-7000 max, excepting the outliers which are seen to reach up to 20,000 in length (with positive sentimental probability of 0.2 and negative sentimental probability of ~0.05)
