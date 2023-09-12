# Classification-of-Fetal-Anatomical-Structures-in-Ultrasound-Images
Developed a Resnet-based transfer learning model to classify fetal anatomical structures in ultrasound images.
# Abstract
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

# Data Preprocessing:
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

