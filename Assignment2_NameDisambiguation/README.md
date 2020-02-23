## 冷启动的姓名消歧
线上学术搜索系统(例Google Scholar, Dblp和AMiner等)已经成为目前全球学术界重要且最受欢迎的学术交流以及论文搜索平台。由于论文分配算法的局限性，现有的学术系统内部存在着大量的论文分配错误；此外，每天都会有大量新论文进入系统。因此，如何准确快速的将论文分配到系统中已有作者档案以及维护作者档案的一致性，是现有的线上学术系统亟待解决的难题.  
**任务**：给定一堆拥有同名作者的论文，要求返回一组论文聚类，使得一个聚类内部的论文都是一个人的，不同聚类间的论文不属于一个人。最终目的给定一个目标名字的论文集，目标是识别出哪些同名作者的论文属于同一个人，返回聚类后的作者cluster，最终评价指标为Macro Pairwise-F1。  
**主要数据**：  
xx可为train, validate, test.  
**xx_author.json**: 作者论文集，除已标注的训练数据集，为一级词典，key为作者名，value为列表储存该作者名下所有的Paper ID.
**xx_pubs.json**: 论文数据集，二级词典，key为Paper ID，value为dict，格式为：
![数据格式示例](https://github.com/k-zha14/WinterCamp/blob/master/Assignment2_NameDisambiguation/dataset_demo.png)

### 前期数据探索
首先，对训练集数据进行一定统计和可视化，从而对数据样本的分布和类型尽心一定的了解。
![训练集文章数](https://github.com/k-zha14/WinterCamp/blob/master/Assignment2_NameDisambiguation/train_data.png)

![训练集作者](https://github.com/k-zha14/WinterCamp/blob/master/Assignment2_NameDisambiguation/train_authors.png)

从上面的两张图可以直观地看出，各作者名下的文章分布极不平衡，在后续处理中需要进行过采样或欠采样处理。以及，**部分作者名下论文为空**，需要对这种边界情况进行考量和处理。

### 模型搭建  
这一部分我主要进行了两方面的工作：1）基于自己的思路进行了尝试；2）在第一部分和WhoIsWho Task1赛道冠军方案的基础上，进行改进增强。最后，在TrainSet进行采样形成验证集，并根据模型在验证集上的表现进行tune，确定超参数的取值，从而确定最终的模型。

#### 基于预训练词向量的有监督迁移学习方案
在深度学习领域，迁移学习已经在各领域取得了成功，从目标检测到图像分割，基本所有高级任务的backbone网络都会现在ImageNet数据集上进行预训练，以提供一个更好的权重初始化点，并加速后续高级任务的训练收敛速度。同样的在NLP领域，先用BERT作为backbone进行预训练也成为了一种业界共识。因此，受这种想法启发，我首先尝试了基于preTrained的词向量进行低级任务的预训练，然后冻结网络参数将网络作为文本特征提取器，移除SoftMax层，将倒数第二fc层的输出做为特征。最后，基于输出的论文特征进行聚类分析。  
在进行上述尝试前，先调试了AMiner官方提供的Baseline模型，获得初步的结果：
 
**Naive Model**|**Baseline 1**|**Baseline 2**|**Baseline 3**
:--:|:--:|:--:|:--:|
0.053|0.121|0.244|0.304

表格中的各模型思路及设置如下: 
* Naive Model: 不做任何预测，认为数据集中的论文均为同一个作者完成;  
* Baseline 1: 按作者所属Org进行规则划分；
* Baseline 2: 按共同作者和所属Org进行规则划分；
* Baseline 3: 以abstract，共同作者，Org为特征，使用TF-IDF组织特征向量，使用DBSCAN进行聚类划分；


#### 基于冠军方案的增强模型
在经过一番尝试后，方案1受计算机限制暂时放弃。借鉴方案1的一些思路和实验结果，我尝试在乔子越团队的冠军方案的机场上进行进一步的改进。原冠军方案的架构如下图所示：
![冠军方案](https://github.com/k-zha14/WinterCamp/blob/master/Assignment2_NameDisambiguation/championPlan.png)
算法的流程大致如下，将输入paper的属性按照是否需要语义理解分为两部分：离散属性（Orgs, Authors, Year, Venue）;语义属性（Title, Abstract, Key Words). 然后分别采用图神经网络表征离散属性的关系结构，使用DeepWalk和Word2Vec抽取获得paper的关系向量（100d）,至于语义属性则直接在语料集上用skip-gram算法训练词向量（100d)，然后将词向量的平均值作为paper语义的平均值。最后将两特征融合（加权和），使用DBSCAN算法进行求解。最后，启发式地定义文本近似算法，对DBSCAN的聚类后的离散点进行再匹配。
上述算法的有一些不足：1）基于DeepWalk和Word2Vec得到的关系表征不鲁棒，且其关联较弱（节点值为：文章ID），这是一种弱映射；
2）特征提取均依赖Word2Vec的无监督训练，文本特征聚合是加和求均值，这种方法对长文本的语义聚合效果很差；  
针对上述不足，我采取了如下改进：  
1）采用Bagging策略，多次DeepWalk，将训练得到的vec求平均，降低特征的噪音；  
2）从前一模型得到启发，训练基于org和联合作者（co-authors)为特征的强关联矩阵分类器，与原来的模型进行Ensemble；
3）Random Search，进行参数优化搜索；

在训练集抽样上进行抽样得到验证集，进过上述改进策略后，结果如下：

**原模型**|**Bagging**|**Ensemble**|**Random Search**
:--:|:--:|:--:|:--:|
0.663|0.672|0.689|0.722

最终，在测试集上的最佳f1-score为
