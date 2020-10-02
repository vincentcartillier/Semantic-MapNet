# Semantic-MapNet

code for the paper:
**[SMNet][1]**
Vincent Cartillier, Zhile Ren, Neha Jain, Stefan Lee, Irfan Essa, Dhruv Batra


website:
[semantic-mapnet.com][2]


## Data
 * ```data/paths.json``` has all the manually recorded trajectories.
 * The semantic meshes with cleaned floor labels will be made available soon.

##Workflow
 * Build training data by running ```make_data/build_training_data_all_features.py``` and ```make_data/build_test_data_projection.py```
 * To train SMNet you can run ```train.py```
 * To evaluate SMNet you can run ```test.py```


## Pre-trained models
 * pretrained models will be made available soon!

[1]: Semantic MapNet: Building Allocentric Semantic Maps and Representations from Egocentric Views
[2]: https://vincentcartillier.github.io/smnet.html


