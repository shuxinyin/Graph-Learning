### 图模型实践
图模型项目实践、论文复现、缓慢更新、欢迎star、交流学习。

#### 1. 环境准备
>pip install -r requirements.txt

#### 2. 数据
>download dataset，put it to ./data/  
  
uploaded dataset blog already

#### 3. 图游走模型：
Note of model deepwalk written here:   
1. [游走图模型--同构图DeepWalk解析](https://zhuanlan.zhihu.com/p/397710211)
2. [游走图模型-聊聊Node2Vec](https://zhuanlan.zhihu.com/p/400849086)

#####  3.1 DeepWalk
①. How to run deepwalk model for graph embedding？
>cd deepwalk
>python main.py

②. node classification
>python node_classification.py


#### Node2Vec
①. How to run Node2Vec model 
>cd node2vec
>python main.py

②. node classification(should changed the checkpoint of node2vec in node_classification.py)
>python node_classification.py
