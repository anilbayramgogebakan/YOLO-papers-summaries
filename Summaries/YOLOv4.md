[link13]: https://github.com/anilbayramgogebakan/YOLO-papers-summaries/blob/main/src/fig13.png
[link13.1]: https://github.com/anilbayramgogebakan/YOLO-papers-summaries/blob/main/src/fig13.1.png
[link14]: https://github.com/anilbayramgogebakan/YOLO-papers-summaries/blob/main/src/fig14.png
[link15]: https://github.com/anilbayramgogebakan/YOLO-papers-summaries/blob/main/src/fig15.png
[link16]: https://github.com/anilbayramgogebakan/YOLO-papers-summaries/blob/main/src/fig16.png
[link17]: https://github.com/anilbayramgogebakan/YOLO-papers-summaries/blob/main/src/fig17.png
[link18]: https://github.com/anilbayramgogebakan/YOLO-papers-summaries/blob/main/src/fig18.png
[link19]: https://github.com/anilbayramgogebakan/YOLO-papers-summaries/blob/main/src/fig19.png
[link20]: https://github.com/anilbayramgogebakan/YOLO-papers-summaries/blob/main/src/fig20.png
[link21]: https://github.com/anilbayramgogebakan/YOLO-papers-summaries/blob/main/src/fig21.png
[link22]: https://github.com/anilbayramgogebakan/YOLO-papers-summaries/blob/main/src/fig22.png
[link23]: https://github.com/anilbayramgogebakan/YOLO-papers-summaries/blob/main/src/fig23.png
[link24]: https://github.com/anilbayramgogebakan/YOLO-papers-summaries/blob/main/src/fig24.png
[link25]: https://github.com/anilbayramgogebakan/YOLO-papers-summaries/blob/main/src/fig25.png
[link26]: https://github.com/anilbayramgogebakan/YOLO-papers-summaries/blob/main/src/fig26.png
[link27]: https://github.com/anilbayramgogebakan/YOLO-papers-summaries/blob/main/src/fig27.png
[link28]: https://github.com/anilbayramgogebakan/YOLO-papers-summaries/blob/main/src/fig28.png

# YOLOv4: Optimal Speed and Accuracy of Object Detection
[Paper](https://arxiv.org/abs/2004.10934) [Github](https://github.com/AlexeyAB/darknet)

*(YOLO, you look only once, but more sharper)*

The main objective was “to optimize neural networks detector for parallel computations”, the developer team also introduces various different architectures and architectural selections after attentively analyzing the effects on the performance of numerous detector, features suggested in the previous YOLO models.

![Figure13][link13]

## Bag of Freebies

Bag of freebies methods are the set of methods that only increase the cost of training or change the training strategy while leaving the cost of inference low.

### 1. Data augmentation
The main objective of data augmentation methods is to increase the variability of an image in order to improve the generalization of the model training. The most commonly used methods are Photometric Distortion, Geometric Distortion, MixUp, CutMix and GANs.

* **Photometric distortion**: Photometric distortion creates new images by adjusting brightness, hue, contrast, saturation and noise to display more varieties of the same image. In the example below we adjusted the Hue (or color appearance parameter) to modify the image and create new samples to create more variability in our training set.

![Figure13.1][link13.1]

* **Geometric distortion**: The so-called geometric distortion methods are all the techniques used to rotate the image, flipping, random scaling or cropping.

![Figure14][link14]

* **MixUp**: Mixup augmentation is a type of augmentation where in we form a new image through weighted linear interpolation of two existing images. We take two images and do a linear combination of them in terms of tensors of those images. Mixup reduces the memorization of corrupt labels, increases the robustness to adversarial examples, and stabilizes the training of generative adversarial networks. In mixup, two images are mixed with weights: λ and 1−𝜆. λ is generated from symmetric beta distribution with parameter alpha. This creates new virtual training samples. In image classification images and labels can be mixed up as following:

![Figure15][link15]

* **CutMix**: CutMix augmentation strategy: patches are cut and pasted among training images where the ground truth labels are also mixed proportionally to the area of the patches. CutMix improves the model robustness against input corruptions and its out-of-distribution detection performances.

![Figure16][link16]

* **Focal loss**: The Focal Loss is designed to address the one-stage object detection scenario in which there is an extreme imbalance between foreground and background classes during training (e.g., 1:1000). Usually, in classification problems cross entropy is used as a loss function to train the model. The advantage of this function is to penalize an error more strongly if the probability of the class is high. Nevertheless, this function also penalizes true positive examples these small loss values can overwhelm the rare class. The new Focal loss function is based on the cross entropy by introducing a (1-pt)<sup>gamma</sup> coefficient. This coefficient allows to focus the importance on the correction of misclassified examples. The focusing parameter γ smoothly adjusts the rate at which easy examples are down-weighted. When γ = 0, FL is equivalent to CE, and as γ is increased the effect of the modulating factor is likewise increased.

![Figure17][link17]

* **Label smoothing**: Whenever you feel absolutely right, you may be plainly wrong. A 100% confidence in a prediction may reveal that the model is memorizing the data instead of learning. Label smoothing adjusts the target upper bound of the prediction to a lower value say 0.9. And it will use this value instead of 1.0 in calculating the loss. This concept mitigates overfitting.

* **IoU loss**: Most object detection models use bounding box to predict the location of an object. To evaluate the quality of a model the L2 standard is used, to calculate the difference in position and size of the predicted bounding box and the real bounding box.The disadvantage of this L2 standard is that it minimizes errors on small objects and tries to minimize errors on large bounding boxes. To address this problem we use IoU loss for the YoloV4 model. Compared to the l2 loss, we can see that instead of optimizing four coordinates independently, the IoU loss considers the bounding box as a unit. Thus the IoU loss could provide more accurate bounding box prediction than the l2 loss. Moreover, the definition naturally norms the IoU to [0, 1] regardless of the scales of bounding boxes

![Figure18][link18]

## Bag of Specials

Bag of special methods are the set of methods which increase inference cost by a small amount but can significantly improve the accuracy of object detection
* **Mish activation**: Mish is a novel self-regularized non-monotic activation function which can be defined by *f(x) = x tanh(softplus(x))*
.
![Figure19][link19]

Why this activation function improve the training?

Mish is bounded below and unbounded above with a range of [≈ -0.31,∞[. Due to the preservation of a small amount of negative information, Mish eliminated by design the preconditions necessary for the Dying ReLU phenomenon. A large negative bias can cause saturation of the ReLu function and causes the weights not to be updated during the backpropagation phase making the neurons inoperative for prediction.

Mish properties helps in better expressivity and information flow. Being unbounded above, Mish avoids saturation, which generally causes training to slow down due to near-zero gradients drastically. Being bounded below is also advantageous since it results in strong regularization effects.

![Figure20][link20]

## Backbone

It uses the CSPDarknet53 as the feature-extractor model for the GPU version. For the VPU(Vision Processing Unit) they consider using EfficientNet-lite — MixNet — GhostNet or MobileNetV3.

The following table shows different considered backbones for GPU version

![Figure21][link21]

Certain backbones are more suitable for classification than for detection. For example, CSPDarknet53 showed to be better than CSPResNext50 in terms of detecting objects, and CSPResNext50 better than CSPDarknet53 for image classification. As stated in the paper, a backbone model for object detection requires Higher input network size, for better detection in small objects, and more layers, for a higher receptive field.

## Neck (detector)

They use Spatial pyramid pooling (SPP) and Path Aggregation Network (PAN). The latter is not identical to the original PAN, but a modified version which replaces the addition with a concat. Illustration shows this:

![Figure22][link22]

Originally in PAN paper, after reducing the size of N4 to have the same spatial size as P5, they add this new down-sized N4 with P5. This is repeated at all levels of 𝑃𝑖+1 and 𝑁𝑖 to produce 𝑁𝑖+1. In YOLO v4 instead of adding 𝑁𝑖 with each 𝑃𝑖+1, they concatenate them (as shown in the image above).

![Figure23][link23]

Looking at the SPP module, it basically performs max-pooling over the 19x19x512 feature map with different kernel sizes k = {5, 9, 13} and ‘same’ padding (to keep the same spatial size). The four corresponding feature maps get then concatenated to form a 19x19x2048 volume. This increases the neck receptive field, thus improving the model accuracy with negligible increase of inference time.

![Figure24][link24]

## Head (detector)

They use the same as YOLO v3.

![Figure25][link25]

## Results

![Figure26][link26]

![Figure27][link27]

![Figure27][link27]

## Useful links:

* Information and tips for Tiny-YOLO: https://github.com/AlexeyAB/darknet/issues/406
* YOLOv4 vs YOLOv5 comparison: https://github.com/AlexeyAB/darknet/issues/5920
* About resizing: https://github.com/pjreddie/darknet/issues/728
