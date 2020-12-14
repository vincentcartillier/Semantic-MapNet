# Semantic-MapNet

Code for the paper:

**[Semantic MapNet: Building Allocentric Semantic Maps and Representations from Egocentric Views][1]**
*Vincent Cartillier, Zhile Ren, Neha Jain, Stefan Lee, Irfan Essa, Dhruv Batra*


Website: [smnet.com][2]

<p align="center">
  <img src='res/img/smnet.png' alt="teaser figure" width="100%"/>
</p>

## Data
 * ```data/paths.json``` has all the manually recorded trajectories.
 * The semantic dense point cloud with cleaned floor labels are available here: https://gatech.box.com/s/enu0tsf9zrog9971iibf591xblhctq7r
 * Ground truth top-down semantic maps are available here: https://gatech.box.com/s/wxzh1bkdtbvccrhyqa4b3fyrjkenu07n

## Workflow
 * To train SMNet you can run ```train.py```
 * To evaluate SMNet you can run ```test.py```


## Pre-trained models
 * pretrained are available here: https://gatech.box.com/s/ap64y9iqxuuwk6fiyd82i3c0ya2w2hah



## License
BDS

[1]: https://arxiv.org/abs/2010.01191
[2]: https://vincentcartillier.github.io/smnet.html
