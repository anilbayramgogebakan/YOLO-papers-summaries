# You Only Look Once: Unified, Real-Time Object Detection

## Introduction

YOLO checks frames only once on the contrary of former models. For example, RCNN use double shot method: First guess the location of possible bounding boxes, then use classification method to decide. Then eliminate duplicate detections. This method decreases the latency and makes YOLO used as real time. Since the model approach to object detection as a regression problem, there are no need to complex pipelines. Despite of high speed, YOLO(v1) was still behind the state-of-art models in terms of accuracy especially small objects.

## Unified Detection

YOLO divide input to SxS grid and process all of the grids parallel.

### Network Design

![Fig1][link1]

Fast YOLO has 9 conv layers, YOLO has 24 conv layers.
### Training

First 20 conv layers pre-trained on the ImageNet, during a week. The accuracy is 88%. For inference, 4 conv and 2 fully connected layers added. Except the final layer, in all the layers leaky activation function has been used. Since most of the bounding boxes does not contain objects, this pushes the confidence score towards zero often overpowering the gradient from cells
that do contain objects. This can lead to model instability. To remedy this, loss from bounding box increased and loss from confidence predictions decreased.

### Inference

For inference and training Darknet framework has been used. As in the training, single network evaluation is used for inference.

### Limitations of YOLO

Because each grid cells predict two boxes and can only have one classes, YOLO model struggles to detect small objects that appear in groups such as flocks of birds. Also, YOLO model learns bounding boxes from data, it struggles to detect objects in different aspect ratios or configurations. 

## Experiments

![Fig2][link2]

![Fig3][link3]

# YOLO9000: Better, Faster Stronger

## Better

Although YOLOv1 is a very fast model compare to other state-of-art models such as Fast-RCNN, recall and localization errors are the bottleneck of YOLOv1. Generally, these problems can be solved by establishing deeper and larger networks. However, these solutions would make the model slower which is the main focus of this architecture. Thus, authors simplify the network and make representation easier to learn.

1. Batch Normalization
   * 2% improvement in mAP

2. High Resolution Classifier
   * After trained by 224×224 images, YOLOv2 also uses 448×448 images
   * 2% improvement in mAP

3. Convolutions with Anchor Boxes
   * YOLOv2 removes all fully connected layers and uses anchor boxes to predict bounding boxes.
   * One pooling layer is removed to increase the resolution of output
   * And 416×416 images are used for training the detection network now. (Slower)
   * Without anchor boxes, the intermediate model got 69.5% mAP and recall of 81%.
   * With anchor boxes, 69.2% mAP and recall of 88% are obtained. Though mAP is dropped a little, recall is increased by large margin.

4. Dimension Clusters
   * In YOLOv1, sizes and scales are pre-defined just like the one in Faster RCNN. In YOLOv2, k-means clustering used which leads to good IOU scores.
![Figure3.1][link3.1]
   * k = 5 is the best value with good tradeoff between model complexity and high recall.
![Figure3.2][link3.2]
   * IOU based clustering with 5 anchor boxes (61.0%) has similar results with the one in Faster RCNN with 9 anchor boxes (60.9%).
   * IOU based clustering with 9 anchor boxes got 67.2%.

5. Direct Location Predictions
   * YOLOv1 does not have constraints on location prediction which makes the model unstable at early iterations. The predicted bounding box can be far from the original grid location.
   * YOLOv2 bounds the location using logistic activation σ, which makes the value fall between 0 to 1:
![Figure3.3][link3.3]

      •	 (cx, cy) is the location of the grid.

      •	(bx, by) is the location of bounding box: (cx, cy) + delta bounded by σ(tx) and σ(ty).

      •	(pw, ph) is the anchor box prior got from clustering.

      •	(bw, bh) is the bounding box dimensions: (pw, ph) scaled by (tw, th).

   * 5% increase in mAP over the version of anchor boxes.

6. Fine-Grained Features
   * The 13×13 feature map output is sufficient for detecting large object.
   * To detect small objects well, the 26×26×512 feature maps from earlier layer is mapped into 13×13×2048 feature map, then concatenated with the original 13×13 feature maps for detection.
   * 1% increase in mAP is achieved.
   * 2% improvement in mAP
 
7. Multi-scale Training
   * For every 10 batches, new image dimensions are randomly chosen.
   * The image dimensions are {320, 352, …, 608}.
   * The network is resized and continue training.

![Figure4][link4]

## Faster

 While most of the models use VGG-16 based architecture, YOLOv2 uses Darknet-19 which is faster than VGG-16. The architecture is below.

![Figure5][link5]

## Stronger

Authors want to train the model with 2 different datasets at the same time: ImageNet and COCO. However, there are some categories that should be put in a hierarchical order. For example, image labelled as “Norfolk Terrier” but it also labelled as “dog” and “mammal”. So, the model show that bounding box as “Norfolk Terrier”. In similar situation, if the model sure that this label is “dog” but not sure about whether is “Norfolk Terrier” or “hunting dog”, it should show as “dog”.

![Figure6][link6]

# YOLOv3: An Incremental Improvement

*(It’s a little bigger but more accurate)*

## Bounding Box Prediction

YOLOv3 also predicts an objectness score(confidence) for each bounding box using logistic regression. This should be 1 if the bounding box prior overlaps a ground truth object by more than any other bounding box prior. For example, (prior 1) overlaps the first ground truth object more than any other bounding box prior (has the highest IOU) and prior 2 overlaps the second ground truth object by more than any other bounding box prior. The system only assigns one bounding box prior for each ground truth object. If a bounding box prior is not assigned to a ground truth object it incurs no loss for coordinate or class predictions, only objectness.

![Figure7][link7]

If the box does not have the highest IOU but does overlap a ground truth object by more than some threshold we ignore the prediction (They use the threshold of 0.5).

## Class (Multi labels) prediction

For some datasets, one image might have more than one labels such as women and person. For this reason, Yolov3 doesn’t use softmax function. Instead it uses binary cross-entropy loss for the class predictions.

## Small objects detection

YOLO struggles with small objects. However, with YOLOv3 we see better performance for small objects, and that because of using short cut connections. Using these connections method allows us to get more ﬁner-grained information from the earlier feature map. However, comparing to the previous version, YOLOv3 has worse performance on medium and larger size objects.

## Feature Extractor Network (Darknet-53)

YOLOv3 uses a new network for performing feature extraction. The new network is a hybrid approach between the network used in YOLOv2 (Darknet-19), and the residual network, so it has some short cut connections. It has 53 convolutional layers so they call it Darknet-53.

![Figure8][link8]

![Figure9][link9]

## Predictions Across Scales

Unlike YOLO and YOLO2, which predict the output at the last layer, YOLOv3 predicts boxes at 3 different scales as illustrated in the below image.

![Figure10][link10]

At each scale YOLOv3 uses 3 anchor boxes and predicts 3 boxes for any grid cell. Each object still only assigned to one grid cell in one detection tensor.

## Performance

When we plot accuracy vs. speed on the AP50 (IOU 0.5 metric), we see that YOLOv3 has signiﬁcant beneﬁts over other detection systems. However, YOLOv3 performance drops signiﬁcantly as the IOU threshold increases (IOU =0.75), indicating that YOLOv3 struggles to get the boxes perfectly aligned with the object, but it still faster than other methods.

![Figure11][link11]

![Figure12][link12]

# YOLOv4: Optimal Speed and Accuracy of Object Detection

*(YOLO, you look only once, but more sharper)*

The main objective was “to optimize neural networks detector for parallel computations”, the developer team also introduces various different architectures and architectural selections after attentively analyzing the effects on the performance of numerous detector, features suggested in the previous YOLO models.

![Figure12][link12]

## Bag of Freebies

Bag of freebies methods are the set of methods that only increase the cost of training or change the training strategy while leaving the cost of inference low.

### 1. Data augmentation
The main objective of data augmentation methods is to increase the variability of an image in order to improve the generalization of the model training. The most commonly used methods are Photometric Distortion, Geometric Distortion, MixUp, CutMix and GANs.

* **Photometric distortion**: Photometric distortion creates new images by adjusting brightness, hue, contrast, saturation and noise to display more varieties of the same image. In the example below we adjusted the Hue (or color appearance parameter) to modify the image and create new samples to create more variability in our training set.

![Figure13][link13]

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