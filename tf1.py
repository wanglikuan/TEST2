import os
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from skimage import data, io, filters
from skimage.color import rgb2gray
from skimage import transform
import random
import datetime

train_data_directory = '/ubda/apps/examples/python/TF/BelgiumTSC_Training/Training'
test_data_directory  = '/ubda/apps/examples/python/TF/BelgiumTSC_Testing/Testing'

def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory) 
                   if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f) 
                      for f in os.listdir(label_directory) 
                      if f.endswith(".ppm")]
        for f in file_names:
            # images.append(skimage.data.imread(f))
            images.append(data.imread(f))
            labels.append(int(d))
    return images, labels

currentDT = datetime.datetime.now()
print ("Starting: ", str(currentDT))

images, labels = load_data(train_data_directory)
images_array = np.array(images)
labels_array = np.array(labels)

# Print the `images` dimensions
print(images_array.ndim)

# Print the number of `images`'s elements
print(images_array.size)

# Print the first instance of `images`
images_array[0]

# Print the `labels` dimensions
print(labels_array.ndim)

# Print the number of `labels`'s elements
print(labels_array.size)

# Count the number of labels
print(len(set(labels_array)))

# Make a histogram with 62 bins of the `labels` data
plt.hist(labels, 62)
# Show the plot
# plt.show()
plt.savefig("disp.png")

# Import the `pyplot` module of `matplotlib`
# import matplotlib.pyplot as plt

# Determine the (random) indexes of the images that you want to see 
traffic_signs = [300, 2250, 3650, 4000]

# Fill out the subplots with the random images that you defined 
for i in range(len(traffic_signs)):
    plt.subplot(1, 4, i+1)
    plt.axis('off')
    plt.imshow(images[traffic_signs[i]])
    plt.subplots_adjust(wspace=0.5)

plt.savefig("disp1.png")
# plt.show()

# Determine the (random) indexes of the images
traffic_signs = [300, 2250, 3650, 4000]

# Fill out the subplots with the random images and add shape, min and max values
for i in range(len(traffic_signs)):
    plt.subplot(1, 4, i+1)
    plt.axis('off')
    plt.imshow(images[traffic_signs[i]])
    plt.subplots_adjust(wspace=0.5)
    # plt.show()
    print("shape: {0}, min: {1}, max: {2}".format(images[traffic_signs[i]].shape, 
                                                  images[traffic_signs[i]].min(), 
                                                  images[traffic_signs[i]].max()))
    
# Get the unique labels 
unique_labels = set(labels)
print("Unique Labels: {0}\nTotal Images: {1}".format(len(set(labels)), len(images)))
# Initialize the figure
plt.figure(figsize=(15, 15))

# Set a counter
i = 1
# For each unique label,
for label in unique_labels:
    # You pick the first image for each label
    image = images[labels.index(label)]
    # Define 64 subplots 
    plt.subplot(8, 8, i)
    # Don't include axes
    plt.axis('off')
    # Add a title to each subplot 
    plt.title("Label {0} ({1})".format(label, labels.count(label)))
    # Add 1 to the counter
    i += 1
    # And you plot this first image 
    plt.imshow(image)
    
# Show the plot
# plt.show()
plt.savefig("disp2.png")
# Resize images
images32 = [transform.resize(image, (28, 28)) for image in images]
images32 = np.array(images32)
# Determine the (random) indexes of the images
traffic_signs = [300, 2250, 3650, 4000]

# Fill out the subplots with the random images and add shape, min and max values
for i in range(len(traffic_signs)):
    plt.subplot(1, 4, i+1)
    plt.axis('off')
    plt.imshow(images32[traffic_signs[i]])
    plt.subplots_adjust(wspace=0.5)
    # plt.show()
    print("shape: {0}, min: {1}, max: {2}".format(images32[traffic_signs[i]].shape, 
                                                  images32[traffic_signs[i]].min(), 
                                                  images32[traffic_signs[i]].max()))
images32 = rgb2gray(np.array(images32))
for i in range(len(traffic_signs)):
    plt.subplot(1, 4, i+1)
    plt.axis('off')
    plt.imshow(images32[traffic_signs[i]], cmap="gray")
    plt.subplots_adjust(wspace=0.5)
    
# plt.show()
plt.savefig("disp3.png")
print(images32.shape)   
 
# sparse_softmax_cross_entropy_with_logits()
# Initialize placeholders 
x = tf.placeholder(dtype = tf.float32, shape = [None, 28, 28])
y = tf.placeholder(dtype = tf.int32, shape = [None])

# Flatten the input data
images_flat = tf.contrib.layers.flatten(x)
# Fully connected layer 
logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)
# Define a loss function
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, 
                                                                   logits = logits))
# Define an optimizer 
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
# Convert logits to label indexes
correct_pred = tf.argmax(logits, 1)

# Define an accuracy metric
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
print("images_flat: ", images_flat)
print("logits: ", logits)
print("loss: ", loss)
print("predicted_labels: ", correct_pred)

tf.set_random_seed(1234)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(201):
        print('EPOCH', i)
        _, accuracy_val = sess.run([train_op, accuracy], feed_dict={x: images32, y: labels})
        if i % 10 == 0:
            print("Loss: ", loss)
        print('DONE WITH EPOCH')

# Pick ren=10 random images
ren=40

sample_indexes = random.sample(range(len(images32)), ren)
sample_images = [images32[i] for i in sample_indexes]
sample_labels = [labels[i] for i in sample_indexes]
# print("images32 ", len(images32),len(sample_images))
# Run the "correct_pred" operation
predicted = sess.run([correct_pred], feed_dict={x: sample_images})[0]                      

# Print the real and predicted labels
print(sample_labels)
print(predicted)

# Display the predictions and the ground truth visually.
fig = plt.figure(figsize=(10, 10))
for i in range(len(sample_images)):
    truth = sample_labels[i]
    prediction = predicted[i]
    # plt.subplot(5, 2,1+i)
    plt.subplot(ren/2, 2,1+i)
    plt.axis('off')
    color='green' if truth == prediction else 'red'
    plt.text(40, 10, "Truth:        {0}\nPrediction: {1}".format(truth, prediction),
             fontsize=12, color=color)
    plt.imshow(sample_images[i],  cmap="gray")

# plt.show()
plt.savefig("disp5.png")

# Load the test data
test_images, test_labels = load_data(test_data_directory )

# Transform the images to 28 by 28 pixels
test_images32 = [transform.resize(image, (28, 28)) for image in test_images]

# Convert to grayscale
test_images32 = rgb2gray(np.array(test_images32))

# Run predictions against the full test set.
predicted = sess.run([correct_pred], feed_dict={x: test_images32})[0]

# Calculate correct matches 
match_count = sum([int(y == y_) for y, y_ in zip(test_labels, predicted)])
# Calculate the accuracy
accuracy = match_count / len(test_labels)
print("Accuracy: {:.3f}".format(accuracy))
sess.close()

currentDT = datetime.datetime.now()
print ("Ended: ", str(currentDT))
