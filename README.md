# WinterCamp
2020 THU WinterCamp by Prof. Tang Jie.  
The whole project is under construction. If any progress is made, it will be pushed as soon as possible.  

## Portal
You can click on the **Assignment Name** and jump to the corresponding website quickly.  
* [**Assignment1 Decision Tree**](https://github.com/k-zha14/WinterCamp/tree/master/Assignment1_DecisionTree)  
> In this project, [Wiscosin Breast Cancer](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)) dataset is loaded. After coding a CART decision tree by hand, I compare the test result with the BenchMark, a [CART](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html) established by scilearn-kit toolbox.   
And the final accuracy is 0.88849(5-fold cv, by Me) V.s 0.91742(scilearn-kit), which is pretty close.
* [**Assignment2 Name Disambigutation**](https://github.com/k-zha14/WinterCamp/tree/master/Assignment2_NameDisambiguation)
> In this part, we focus on disambigutating the phenomena, that many authors share the same name but are not the same person.  For example, someone, called Li Hua from Tsinghua University, may study on Deep Learning. And he have successfully published a lot papers on ICCV, CVPR, ECCV and etc. However, there are some other Li Huas, studying on other fields and publish other papers. So it is very necessary to solve the problem for Google Scholar, Aminer and other scholar platforms. We apply Heterogeneous Graph to establish the knowledge graph and Word2Vec to map word to dense vector. After extracting segmantic vector from papers, we select DBSCAN to cluster different authors.   
Finally, this method achieve 0.4662 macro-pairwise F1 score in test dataset.
