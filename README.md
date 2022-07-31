This is a repository for Barchelor's degree project at KFU. The thesis is:

# Self-supervised learning for semantic segmentation
Training deep learning models requires lots of training data. Labelling data can become expensive in terms of time and money especially for computer vision tasks such as semantic segmentation. To overcome this issue several self-supervised learning methods have been proposed that utilize only images without labels. Most of the work has been done on the self-supervised learning at the image obejct level, but there are computer vision tasks that require more dense feature representations (e.g. semantic segmentation, object detection). Thus some methods for dense self-supervised feature learning have been proposed and one of them is the basis of this work: [Propagate Yourself: Exploring Pixel-Level Consistency for Unsupervised Visual Representation Learning]("https://arxiv.org/pdf/2011.10043.pdf "pixpro").

# Method


The goal of this work was to adapt the proposed dense self-supervised learning method for semantic segmentation task. The key modification was increasing resolution of the output feature maps to have the same height and width as the input image thus preserving feature vector for every pixel. Pixel vector features are then used in linear evaluation by adding a logistic regression over it. If the pixel representations are good enough they can be easily separated and classified by the linear layer.

![PixPro Modification Overview](/pixpro_files/img/pixpro_arch.PNG)

Backbone network architecture was changed from ResNet-50 to UNet to keep the same resolution as the input image.

To make the larger resolution of the output feature maps possible, few modifications were made to be able to fit the available memory:

- Positive Pixel Pairs

Instead of computing the distances between every pixel of one feature map and every pixel of another feature map and then choosing positive pairs based on the computed distances, this work computes distances only for a small subset of pixels from the intersection of the two feature maps (as close pixels are sure to lie near intersection or to be in the intersection of two views). Distances are then used to form positive pairs and only a small subset is yet again taken.

![Positive Pixel Pairs (blue for pixels from the first view, red for pixels of the second view, green for pixels that intersect)](/pixpro_files/img/pos_pairs.PNG)

All the following methods including positive pixel pairs are then modified to work only with those positive pairs.

- Local PPM

Pixel-Propagation Module proposed in the original work includes calculating similarity between pixel vector representations between every pixel pair of the feature map which is ambigious. To overcome this issue only local regions are used for calculating similarity and propagating features. This is motivated by the fact that pixels in local regions are more likely to have the same semantics.

![Local PPM](/pixpro_files/img/ppm.png)

# Experiments
Training and linear evaluation on semantic segmentation dataset Cityscapes

# Visual representation
Results of pixel retrieval

