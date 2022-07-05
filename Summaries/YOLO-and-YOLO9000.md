[link1]: https://github.com/anilbayramgogebakan/YOLO-papers-summaries/blob/main/src/fig1.png
[link2]: https://github.com/anilbayramgogebakan/YOLO-papers-summaries/blob/main/src/fig2.png
[link3]: https://github.com/anilbayramgogebakan/YOLO-papers-summaries/blob/main/src/fig3.png
[link3.1]: https://github.com/anilbayramgogebakan/YOLO-papers-summaries/blob/main/src/fig3.1.png
[link3.2]: https://github.com/anilbayramgogebakan/YOLO-papers-summaries/blob/main/src/fig3.2.png
[link3.3]: https://github.com/anilbayramgogebakan/YOLO-papers-summaries/blob/main/src/fig3.3.png
[link4]: https://github.com/anilbayramgogebakan/YOLO-papers-summaries/blob/main/src/fig4.png
[link5]: https://github.com/anilbayramgogebakan/YOLO-papers-summaries/blob/main/src/fig5.png
[link6]: https://github.com/anilbayramgogebakan/YOLO-papers-summaries/blob/main/src/fig6.png

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
