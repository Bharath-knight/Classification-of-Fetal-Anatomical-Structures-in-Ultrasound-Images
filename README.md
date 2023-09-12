# Classification-of-Fetal-Anatomical-Structures-in-Ultrasound-Images
Developed a Resnet-based transfer learning model to classify fetal anatomical structures in ultrasound images.
## Abstract
A deep learning model was developed to address the difficult task of classifying
anatomical structures in ultrasound images. The ResNet50 architecture, a deep
convolutional neural network that has been demonstrated to be effective for image
classification tasks, was used. To improve its performance, the model was trained on a
dataset of 500 ultrasound images and their labels, and data augmentation techniques
were used. On the validation set, the model had an accuracy of 85%. It could, however,
be improved further by using a larger dataset and a more sophisticated model
architecture. Other hypotheses, such as using a different model architecture, optimizer,
or loss function, were also investigated by myself. These experiments contributed to a
better understanding of the factors that influence the model's performance.

## Data Preprocessing:
The first step in the project was to preprocess the data. This involved resizing the images
to a standard size, normalizing the images, and converting the labels to one-hot encoded
vectors.
All the images were present in different sizes and I had resized the images to (661, 661,3)
as I read from the literature that using a transfer learning approach, yields higher
accuracy when the images are increased in size while inputting to the model.
The images were normalized by subtracting the mean and dividing by the standard
deviation of the training set. This normalization step helps to improve the performance of
the model.
The labels were converted to one-hot encoded vectors. This means that each label was
represented as a vector that had a value of 1 for the corresponding class and a value of 0
for all other classes.
The augmentation steps that I used were resizing the images to a standard size,
Normalizing the images, Randomly flipping the images horizontally and vertically,
Randomly rotating the images, and Randomly cropping the images.

## Model Architecture:
ResNet50 comprises 50 convolutional layers organized into 4 stages, each containing
multiple residual blocks. A residual block consists of a pair of convolutional layers. The
initial layer's output is combined with the subsequent layer's output, enabling the model
to grasp residual connections. These connections serve as shortcuts that circumvent
certain layers within the network.


![Resnet_50_Architecture](https://github.com/Bharath-knight/Classification-of-Fetal-Anatomical-Structures-in-Ultrasound-Images/assets/82858787/2a094a54-3144-4af8-8ab2-5feed09abfd3)

I opted for ResNet50 as my model of choice due to its distinct advantages, particularly in
the realm of medical imaging. As a deep model, it boasts an extensive parameter count,
empowering it to capture intricate data patterns effectively. The incorporation of skip
connections acts as a safeguard against excessive depth, mitigating the risk of overfitting
on training data. Moreover, its pre-training on an expansive image dataset provides foundational knowledge for classifying ultrasound images.

## Experimental Setting:
### Loss function:
The loss function used was categorical crossentropy. Categorical
crossentropy is a loss function that is used for classification tasks. It measures the
difference between the predicted probabilities and the ground truth labels.
### Learning rate:
The learning rate was set to 0.01. The learning rate controls how much the
model's parameters are updated during training. A lower learning rate will result in a
more gradual update of the parameters, while a higher learning rate will result in a more
aggressive update of the parameters.
### Learning rate scheduler:
The learning rate scheduler used was ReduceLROnPlateau.
This helps to prevent the model from overfitting the training data. I went ahead with the
common learning rate decay schedule where the learning rate decayed by 0.2 every 2
epochs i.e. the learning rate was multiplied by 0.8 after every 2 epochs.
### Batch size: 
The batch size is the number of images that are processed at a time during
training. I tried to use a larger batch size (eg:16,32) to help to improve the performance of
the model, but it was computationally expensive and I wasn’t able to run my model with
my computing resources.
### Number of epochs:
The number of epochs is the number of times that the model is
trained on the entire training dataset. I tried going with a higher number of epochs (50)
which would have resulted in a more accurate model, but again I wasn’t able to run my
model with my computing resources so I went with 10 epochs.

## Hypothesis tried:
While ResNet50 is undeniably a robust and efficient choice. I implemented the VGG19 model
to compare the accuracy with this model and I found the trainable parameters in VGG19
were higher hence, the training data was overfitted for the model.
Each optimizer brings its own set of merits and drawbacks to the table. I tried my
implementation with optimizers like Adam and found that Stochastic Gradient Descent
was most suitable for this classification task.
Furthermore, considering loss functions, I tried implementing binary crossentropy as
an alternative to the widely-used categorical crossentropy loss function. I also finetuned
parameters to achieve a higher accuracy score.
This experimentation deepened my understanding of the intricacies of model training
and helped me to make more informed decisions.

## Future Work:
With computing resources, I would:
● Be using a larger dataset of images which will improve the performance of the
model
● Increasing the number of epochs will improve the performance of the model.
● Regularizing the model will improve the performance of the model and reduce
overfitting

