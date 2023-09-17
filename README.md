# Starbucks-Customer-Segmentation
Customer segmentation for Starbucks reward program members on the basis of their reactions to offers, to further design marketing strategies accordingly 

I watched the netflix documentary __"The Social Dilemma"__ and got fascinated by how marketing strategies use Data Science in our era of Big Data.

I also read an article on how __Starbucks is like a bank__ on Groww ( yeah, me proud reader of Groww and Finshots), and found it intresting how starbucks earns lot of money through intrest earned on the money deposited. Being a curious person, I decided to investigate how starbucks convinced its customers to deposit so much money with them. That's how I discovered Starbuck's reward program. 

So, being a coffee lover and someone intrested in business mechanisms, I decided to take up this project. (My favourite is Caramel Latte, Thanks :) 

![image](https://github.com/ayush-agarwal-0502/Starbucks-Customer-Segmentation/assets/86561124/db204c3c-5cdf-421c-a00c-ae3a7edb1287)
![image](https://github.com/ayush-agarwal-0502/Starbucks-Customer-Segmentation/assets/86561124/f259c730-3e60-49b1-96c4-4855bdcbb8d8)



__Q Why segment customers ???__

Ans - Well, because it is still a very difficult and computationally intensive task to target each customer in unique ways, as well as there would be lots of customers on which similar Marketing strategies and Discount offers would work well. Hence it makes intutive sense to break the customer base into clusters on the basis of how and which offers they react to and then proceed to make targeted strategies. 

## If I had a minute to tell u about this project - 

* Coffee, Business, Marketing, Data Science , Action !
* Performed __Data Engineering__ to merge the 3 datasets, performed __Data Cleaning__ (Some enteries were errors cause their data was missing as well as age was 118 which seemed to be an error).
* Used __Power Transforms__ to make dataset more gaussian since __K-means__ likes rounder (isotropic) clusters. 
* Performed __PCA ( Principal Component Analysis )__ to __reduce Data Dimensionality__ and avoid __Curse of Dimensionality__ , and also so that redundant features would be removed and the data would be in continuous so that we don't need to bother ourselves with __K-Prototype__ kind of algorithms ( its K-means plus K-mode for mixed (continuous plus categorical) data modelling ) 
*  Used __K-means clustering__ to form clusters, along with __Silhouette score__ and __SSE(Sum of Squared Errors)/Inertia__ to determine optimal number of clusters, and then __Silhouette Plots__ to further analyse quality of clusters.
* Visualized Clusters in 2-D using __t-SNE__ to further confirm the feasiblity of clusters.
* Performed __Gaussian Mixture Modelling__ since it is a __Generative__ and __not Discriminative classifier__, also it's a __soft classifier__, to allow further scope for data-points on borders of clusters or customers which behave like 2 or more clusters. Also it can form "non-round" clusters better too. 

