## _Upgrading Pipeline_

### _1. Upgrade Code_

Upgrate code to the latest version of PyTorch Lightning.


### _2. Name of Stages_

We will rename the each stage to reflect the functionality of these stages in the pipeline. For example,

1. _`data_processing`_
2. _`edge_construction` OR `graph_contruction`_
3. _`edge_filtering` OR `graph_filtering`_
4. _`edge_labelling` OR `graph_labelling`_
5. _`graph_segmenting`_


### _3. Output of Stages_

The output directories can be fixed in config files to reflect the name of each stages:

1. _`data_processing` OR `feature_store`_
2. _`edge_construction` OR `graph_contruction`_
3. _`edge_filtering` OR `graph_filtering`_
4. _`edge_labelling` OR `graph_labelling`_
5. _`graph_segmenting`_


**NOTE**: _**edge labelling**_ or _**graph labelling**_ stage is infact **_edge classification_** stage of the pipeline.


### _4. Stage Improvements_

#### _4.1 Graph Segmenting_

Improve graph segmenting stage:

- make graph segmenting useful
- output of segmentation stage should be _`run/graph_segmenting/test`_ to indicate that it is _`test`_ dataset from previous stage
- add usefull parameters for each track building method e.g. DBSCAN need _`epsilon`_, CCL mehtod need _`edge_cut`_
- DBSCAN requires optimal _`epsilon`_ usually set as `0.25`, CCL required optimal _`edge_cut`_ usually set as `0.5`

Thats where `graph_segmenting` stage ends.


#### _4.2 Track Evaluation_

- _Source code for track evaluation is located in _`eval/`_ directory_
    - _One can also run `graph_segmenting` stage or `trkx_from_gnn.py` (rename it as `track_building.py`) from here as well_
- _Output of track evaluation is located in _`run/graph_segmenting` or `run/track_evaluation`__


Two main task once track building or graph segmenting is done are:

1. _`eval_reco_trkx.py` or rename it as `track_evaluation.py`_
2. _`plot_trk_perf.py` or rename it as `track_performace.py`_

Update script names if renamed in the README files.
___

```shell
# handle import error
try:
    # if release 6+ imports not available
    from skim import BaseSkim, CombinedSkim
    from skim.WGs.ewp import BtoXll
except (ImportError, ModuleNotFoundError):
    # then usee release 5 imports
    from skimExpertFunctions import BaseSkim, CombinedSkim
    from skim.ewp import BtoXll
```


