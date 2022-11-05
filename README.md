# AID4HAI
Automatic Idea Detection for controlling Healthcare Associated Infections

# Code
Please extract tweets' text using tweets' ID and add it to the dataframe before running this code

The code is based on RoBERTa model from Huggingface’s transformers library. https://huggingface.co/docs/transformers/model_doc/roberta

For more information on how to install the required libraries, please visit: https://towardsdatascience.com/tensorflow-and-transformers-df6fceaf57cc

The following picture demostrates the architecture of the model:
![Alt text](images/model.png?raw=true "Transfer learning using BERTweet language model")

![Alt text](images/experiment1.png?raw=true "Transfer learning using BERTweet language model")
![Alt text](images/experiment2.png?raw=true "Transfer learning using BERTweet language model")


# Dataset

AID4HAI dataset is a collection of Tweets related to Controling Healthcare-Associated Infections.

AC and BC in the file names refere to tweets extracted after and before covid pandemic.

Each tweet is labeled by three anotators indicating whether it includes an idea about Control Healthcare-Associated Infections or not.

Tweets suggesting an idea are marked by 1 others by label 0.

Before using this dataset you need to extract the tweet's text using  Twitter developer account.

An easy way to extract the tweets information is to place it at the end of the following link: https://twitter.com/bramus/status/<twitter_id> e.g. https://twitter.com/bramus/status/932586791953158144

![Alt text](images/table1.png?raw=true "table1" width="500")
![Alt text](images/table2.png?raw=true "table2")
![Alt text](images/table3.png?raw=true "table3")
![Alt text](images/table4.png?raw=true "table4")
![Alt text](images/table5.png?raw=true "table5")
![Alt text](images/table6.png?raw=true "table6")
![Alt text](images/table7.png?raw=true "table7")

# Conclusion

Following is a list of automatically extracted ideas related to HAI from twitter:
![Alt text](images/ideas.png?raw=true "idea")

# Citation
If you are using the code or the data shared in this repository, please cite the following paper:

!fill in paper citation information!
