## _Upgrading Pipeline_

### _Upgrade Code_

Upgrate code to the latest version of PyTorch Lightning.


### _Renaming of Stages_

We will rename the each stage to reflect the functionality of these stages in the pipeline. For example,

1. _`Processing`_ stays
2. _`Construction`_ $\leftarrow$ _`Embedding`_
3. _`Filtering`_ stays
4. _`Labelling`_ $\leftarrow$ _`GNN`_
5. _`Segmenting`_ stays


Or, one rename them as follows

1. _`data_processing`_ $\leftarrow$ _`Processing`_ 
2. _`edge_construction`_ or _`graph_contruction`_ $\leftarrow$ _`Embedding`_
3. _`edge_filtering`_ $\leftarrow$ _`Filtering`_
4. _`edge_labelling`_ $\leftarrow$ _`GNN`_
5. _`graph_segmenting`_ $\leftarrow$ _`Segmenting`_  


### _Output of Stages_

The output directories can be fixed in config files to reflect the name of each stages:

1. _`data_processing`_ or _`feature_store`_
2. _`edge_construction`_ or _`graph_contruction`_ $\leftarrow$ _`embedding_processed`_
3. _`edge_filtering`_ $\leftarrow$ _`filter_processed`_
4. _`edge_labelling`_ $\leftarrow$ _`gnn_processed`_
5. _`graph_segmenting`_ $\leftarrow$ _`segments_processed`_


**NOTE**: _**edge labelling**_ or _**graph labelling**_ stage is infact **_edge classification_** stage of the pipeline.