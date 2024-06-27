# Computer Vision - Project 2

## Description

From a sample of images containing lego pieces, we want to be able to create a program that can detect:

- The number of lego pieces
- The position of each lego piece in the image using bounding boxes and segmentation

This should be done using deep learning techniques such as CNNs.

## Dataset

As the team had started the second phase of the project far before the providing of dataset split by the teachers, we developed our own based on the original full 5.6GB dataset.

In a nutshell, the team took all 5841 lego images (with 5300 being of single lego pieces which is relevant due to dataset imbalance) and rescaled them into a normalized 224x224 size.

All models used in task 2 used a 70% (training) + 30% (validation/testing) split for their fine-tuning and 80% + 20% in task 3.

## Lego Counting

The purpose of this task was to create a model that could predict how many legos were in an image using a CNN. The team explored different possibilities in order to develop a capable model.

The first approach was to implement a classification model with 32 output nodes (image maximum number of legos is 32). Unfortunately, due to the imbalance of the dataset, this model only predicted that images had a single lego.

As a way to solve this problem, the team used data augmentation to increase the number of images that had more than one lego piece by performing flip and rotation operations.

Despite this, we also decreased the number of images that contained a single lego, since its proportion was still too high.

During training, we noticed that in this task, the learning rate actually had a very big impact on the results, and sometimes a minor change to it, could lead to drastic changes to the model. Even with a learning rate of 0.001 the model didn’t converge most of the time, and hence, needed to be adjusted, even while using the Adam optimizer.

The next step was to switch from using classification to using regression instead. However, the regression models didn’t perform well, when compared to the classification models.

Finally, we increased the number of epochs done during the training of the models and satisfying results were finally achieved. In order to access our results, 3 different models were trained using both classification and regression. The first model was a simple custom CNN, which was mostly used as a control, and produced the worst results. The other two, were pre-trained models, VGG16 and ResNet18, which achieved significantly better results.

## Lego Detection

The team explored, for the purpose of lego detection using bounding boxes, two different already pre-trained CNN models: `YOLOV8` from *Ultralytics* and `Faster R-CNN`.
The models were fine-tuned on our dataset for 5 epochs (due to time and hardware constraints) with both models requiring very particular dataset input formatting and configuration that were very challenging to achieve.

### Lego Segmentation

We explored many alternatives, including applying CNNs to each bounding box to extract the foreground. The CNN approach is quite expensive in terms of both memory and computational resources and the results were very poor because we didn’t have a way to train them due to lack of ground truth masks. The team experimented with pre-trained `U-Net` and `Mask R-CNN`, with the first yielding unusable masks almost 100% of the time.

The second fared a bit better, but still failed to be consistent by sometimes not detecting anything as foreground or giving badly defined lego shapes. In 30 bounding box images, Mask R-CNN only gave masks different than blank to less than a third, making it an unreliable solution.

## Group Members

|      Name      | Student Number |
| -------------- | -------------- |
| João Alves     |   202007614    |
| Marco André    |   202004891    |
| Rúben Monteiro |   202006478    |
