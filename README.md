# Semantic-MapNet

Code for the paper:

**[Semantic MapNet: Building Allocentric Semantic Maps and Representations from Egocentric Views][1]**
*Vincent Cartillier, Zhile Ren, Neha Jain, Stefan Lee, Irfan Essa, Dhruv Batra*


Website: [smnet.com][2]


## Data
 * ```data/paths.json``` has all the manually recorded trajectories.
 * The semantic meshes with cleaned floor labels will be made available soon.

## Workflow
 * Build training data by running ```make_data/build_training_data_all_features.py``` and ```make_data/build_test_data_projection.py```
 * To train SMNet you can run ```train.py```
 * To evaluate SMNet you can run ```test.py```


## Pre-trained models
 * pretrained models will be made available soon!



## License
BDS

[1]: https://arxiv.org/abs/2010.01191
[2]: https://vincentcartillier.github.io/smnet.html
