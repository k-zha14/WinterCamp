# Project 1: Decision Tree
This experiment is roughly devided into 3 sections, Data Understanding, Building CART(Classification and Regression Tree) and Experiment. By the way, these sections are corresponding with code in main.py.
## Data Understanding
From [**Site**](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)), we can easily get a general understanding of the whole dataset through calling pandas.DataFrame().info(). Some basic infomation are listed as follows.  
* number of classes: 2
> *m* means malignant, corresponding 0 in dataset;  
> *b* means benign, corresponging 1 in dataset;  
* number of samples: 569  
> *m* occupy 212, *b* occupy 357
* number of featuress: 30  
> inluding *mean radius, mean texture, mean perimeter* and etc. These are all continuous real values, and the range of these features are varied vastly form each other.  

Luckily, there are no decision tree is used to solve this classification problem and there are no missing values in original dataset. So there are no necessary to do further data preprocess.

## Build CART
Decision Tree can be roughly devided into three groups, C4.5, ID3 and CART. Considering the feature values are continuous, CART tree is more suitable. The progress of building a CART decision tree is illustrated as follows.  
Assuming there is dataset *D* of node *p*, the size of *D* is k\*30(2-D matrix)
* Step 1, Search the best segment feature and the best seperate point according to Gini loss.
> ![Gini loss](https://shuwoom.com/wp-content/uploads/2018/10/179aa82268e3c3b3fb17a60adb545bb8.png "Gini loss")  
where, *A* represents one of the freatures; *D1* and *D2* is the devided sub-datasets. Through comparision of gini loss, we select the minimum choice, which is corresponding to best segment feature and the best seperate point. 
* Step 2, devide dataset *D* to two sub-datasets *D1* and *D2* and transfer these to the children nodes of *p*
* Lastly, recursively call step 1 and step 2.

## Experiment  
According to homework assignments, 5-fold cross-validation is used to train and test the CART decision tree. What's more, I compare my CART with sklearn.tree.DecisionTreeClassifier(as Benchmark). And the results are close.

What needs to emphasize is that some stop conditions should be set to avoid overfitting. After several attempts, I choose ***gini_loss>1e-3***, ***max depth of decision tree<3*** and ***samples>10*** as stop condition. 

My decision tree:  

1st-fold|2nd-fold|3st-fold|4th-fold|5th-fold|Avg Acc
--|:--:|--:|--:|--:|--|
78.76%|85.84%|96.46%|93.80%|89.38%|88.84%

**(The original results can be seen in cell 16 of main.ipynb)**

And the Benchmark is 91.74%. In comparision, the results are pretty close.

