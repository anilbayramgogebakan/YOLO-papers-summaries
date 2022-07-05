[link7]: https://github.com/anilbayramgogebakan/YOLO-papers-summaries/blob/main/src/fig7.png
[link8]: https://github.com/anilbayramgogebakan/YOLO-papers-summaries/blob/main/src/fig8.png
[link9]: https://github.com/anilbayramgogebakan/YOLO-papers-summaries/blob/main/src/fig9.png
[link10]: https://github.com/anilbayramgogebakan/YOLO-papers-summaries/blob/main/src/fig10.png
[link11]: https://github.com/anilbayramgogebakan/YOLO-papers-summaries/blob/main/src/fig11.png
[link12]: https://github.com/anilbayramgogebakan/YOLO-papers-summaries/blob/main/src/fig12.png

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
