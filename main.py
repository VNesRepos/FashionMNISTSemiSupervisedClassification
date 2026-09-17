import tensorflow as tf
import numpy as np
import matplotlib as plt



# Stage 1: Data Preparation and Preprocessing

print("Loading...")

data = np.genfromtxt('fashion-mnist_train.csv', delimiter=',', skip_header=1)

# everything in column 0 is a label and goes to labels
labels = data[:, 0].astype(int)
# everything in columns 1 to 784 are grayscale pixels that are placed in the pixels matrix
# that represents a fashion data sample  of size 28 x 28
pixelsAsAttributes = data[:, 1:]

# Z-Score Normalization
# every column in pixelsAsAttributes represents the same pixel in the 28 x 28 pixels
# calculate the mean for each column of pixelsAsAttributes
mean = np.mean(pixelsAsAttributes, axis=0)
#calculate the standard deviation per column of pixelsAsAttributes
std = np.std(pixelsAsAttributes, axis=0)
#use the mean and standard deviation to normalize the data across each of their respective columns
pixelsAsAttributes = (pixelsAsAttributes - mean) / (std + 1e-7)

# 0-1 normalization
"""p_min = np.min(pixelsAsAttributes, axis=0)
p_max = np.max(pixelsAsAttributes, axis=0)
# for each value in the pixelsAsAttributes matrix we subtract it from the lowest
# value in its respective column and divide the result by the difference between
#the smallest and the biggest value in the corresponding column
# add the epsilon (1e-7) to avoid division by zero
pixelsAsAttributes = (pixelsAsAttributes - p_min) / (p_max - p_min + 1e-7)
"""
# Shuffle and split data
#indices = row indexes used to perform shuffle the rows/samples/vectors of 784 attributes
# before the split assignment
indices = np.arange(len(pixelsAsAttributes))
#key seed that allows for recreating the results
np.random.seed(1)
#perform the shuffle with the random seed
np.random.shuffle(indices)

#count the number of rows
count = len(pixelsAsAttributes)
#number of samples that will be used for testing/validation
testCut = int(count * 0.10)
validationCut = testCut + int(count * 0.10)

#slice for testing
testingDataX = pixelsAsAttributes[indices[:testCut]]
testingDataY = labels[indices[:testCut]]

#slice for validation
val_X = pixelsAsAttributes[indices[testCut:validationCut]]
val_y = labels[indices[testCut:validationCut]]

#whatever isn't sliced into testing or validation matrices
# goes into training
trainingDataX = pixelsAsAttributes[indices[validationCut:]]
trainingDataY = labels[indices[validationCut:]]

# Semi-supervised split: 20% Labeled
#get the number of rows in trainingData
n_pool = len(trainingDataX)
labeled_split = int(n_pool * 0.5)
#20% slice for labeled data
labeledX = trainingDataX[:labeled_split]
labeledY = trainingDataY[:labeled_split]
#the rest 80% goes to unlabeledData
unlabeledX = trainingDataX[labeled_split:]
#1.augmentation method that turns the row/vectors back into their original 28x28 format
#2.then flips the image in its y-axis(horizontally)
#3. turns the 28 x 28 back into a 1 X 784 vector
#4. using Gaussian distribution generate and adds the noise
#5. cap the weights at -5,+5 to prevent exploding gradients
def flip_and_noise(X_batch):
    reshaped = X_batch.reshape(-1, 28, 28)
    flipped = reshaped[:, :, ::-1]
    X_aug = flipped.reshape(-1, 784)
    noise = np.random.normal(0, 0.15, X_aug.shape)
    X_aug = X_aug + noise
    return np.clip(X_aug, -5, 5)

# Augmentation and concatenation with the original
LabeledLugmentedX = flip_and_noise(labeledX)
combinedSamplesX = np.concatenate((labeledX, LabeledLugmentedX), axis=0)
combinedSamplesY = np.concatenate((labeledY, labeledY), axis=0)

#shuffle the concatenated samples
shuffle_idx = np.arange(len(combinedSamplesX))#getLength
np.random.shuffle(shuffle_idx)#get randomIndeces
#do the shuffle
trainDataFinalX = combinedSamplesX[shuffle_idx]
trainDataFinalY = combinedSamplesY[shuffle_idx]

# Random Uniform initialization
def resetWeights():
    # using 'global' to update the variables defined outside this function
    global W1, b1, W2, b2, W3, b3, variables, inertia

    # W1 weight matrix between inputLayer(sample vector), 784 and hidden Layer 1(512 neurons)
    lim1 = 1.0 / np.sqrt(784)
    W1 = tf.Variable(tf.random.uniform([784, 512], minval=-lim1, maxval=lim1), name="W1")
    b1 = tf.Variable(tf.zeros([512]), name="b1")

    # W2 weight matrix between hidden Layer 1,( 512 neurons) and hidden Layer 2(256 neurons)
    lim2 = 1.0 / np.sqrt(512)
    W2 = tf.Variable(tf.random.uniform([512, 256], minval=-lim2, maxval=lim2), name="W2")
    b2 = tf.Variable(tf.zeros([256]), name="b2")

    # W3 weight matrix between hidden Layer 2( 256 neurons) and output Layer, 10 neurons = number of classes
    lim3 = 1.0 / np.sqrt(256)
    W3 = tf.Variable(tf.random.uniform([256, 10], minval=-lim3, maxval=lim3), name="W3")
    b3 = tf.Variable(tf.zeros([10]), name="b3")

    variables = [W1, b1, W2, b2, W3, b3]
    inertia = [tf.zeros_like(v) for v in variables]



"""# Using fixed small standard deviation (0.01)
W1 = tf.Variable(tf.random.normal([784, 512], stddev=0.01), name="W1")
b1 = tf.Variable(tf.zeros([512]), name="b1")

W2 = tf.Variable(tf.random.normal([512, 256], stddev=0.01), name="W2")
b2 = tf.Variable(tf.zeros([256]), name="b2")

W3 = tf.Variable(tf.random.normal([256, 10], stddev=0.01), name="W3")
b3 = tf.Variable(tf.zeros([10]), name="b3")

variables = [W1, b1, W2, b2, W3, b3]
inertia = [tf.zeros_like(v) for v in variables]"""



def modelForward(x, training=True, dropout_rate=0.5):
    # Layer 1: Matrix multiplication and Sigmoid activation
    z1 = tf.matmul(x, W1) + b1
    a1 = tf.sigmoid(z1)

    if training:
        mask1 = tf.cast(tf.random.uniform(a1.shape) > dropout_rate, tf.float32)
        a1 = (a1 * mask1) / (1.0 - dropout_rate)  # Scale to keep signal strength

    # Layer 2: Matrix multiplication and Sigmoid activation
    z2 = tf.matmul(a1, W2) + b2
    a2 = tf.sigmoid(z2)

    if training:
        mask2 = tf.cast(tf.random.uniform(a2.shape) > dropout_rate, tf.float32)
        a2 = (a2 * mask2) / (1.0 - dropout_rate)

    # Output Layer (Logits)
    return tf.matmul(a2, W3) + b3

def computeLoss(y_true, logits):
    # Convert logits to probability distribution
    probs = tf.nn.softmax(logits)

    # Create one-hot mask for correct labels
    y_true_one_hot = tf.one_hot(y_true, depth=10)

    # Select probabilities of correct labels
    correct_class_probs = tf.reduce_sum(y_true_one_hot * probs, axis=1)

    # Negative log likelihood of correct probabilities
    loss = -tf.reduce_mean(tf.math.log(correct_class_probs + 1e-8))

    return loss



def trainNN(X_train, y_train, X_val, y_val, lr, alpha, batch_size):
    #dictionary to store validation loss and accuracy over epochs
    history = {'val_loss': [], 'val_acc': []}
    #get the total number of rows/samples in the training dataset
    num_samples = X_train.shape[0]
    #set the learning rate that can potentially be changed during training
    adaptive_lr = lr

    #loop through a maximum of 100 training cycles (epochs)
    for epoch in range(100):
        # Shuffle indices at start of epoch
        #generate a random sequence of row indexes from 0 to num_samples
        indices = np.random.permutation(num_samples)
        #use the random indexes to shuffle the training features
        XShuffle = X_train[indices]
        #use the exact same random indexes to shuffle the corresponding labels
        YShuffle = y_train[indices]

        """if epoch > 0 and epoch % 2 == 0:
            adaptive_lr *= 0.9
            print(f"--- Learning rate dropped to {adaptive_lr} ---")"""

        #loop through the shuffled data in chunks of batch_size
        for i in range(0, num_samples, batch_size):
            #slice out a batch of features and cast them to float32 for TensorFlow
            x_batch = tf.cast(XShuffle[i:i + batch_size], tf.float32)
            #slice out the corresponding batch of labels and cast to int32
            y_batch = tf.cast(YShuffle[i:i + batch_size], tf.int32)

            # Record gradient information
            #open a GradientTape to track operations for backpropagation
            with tf.GradientTape() as tape:
                #1. do a forward pass to get the predictions for the current batch
                preds = modelForward(x_batch, training=True)
                #2. calculate the loss using the predictions and the actual labels
                loss = computeLoss(y_batch, preds)

            # Perform backpropagation
            #calculate the gradients of the loss with respect to all network variables (weights/biases)
            grads = tape.gradient(loss, variables)

            # Manual SGD update with momentum
            #loop through every variable (W1, b1, W2, b2, W3, b3)
            for j in range(len(variables)):
                #calculate the new momentum/inertia based on the alpha factor and the current gradient
                inertia[j] = alpha * inertia[j] + grads[j]
                #update the actual weight/bias variable by subtracting the adjusted learning step
                variables[j].assign_sub(adaptive_lr * inertia[j])

        # Validation evaluation
        #do a forward pass on the entire validation dataset (training=False turns off dropout)
        val_preds = modelForward(tf.cast(X_val, tf.float32), training=False)
        val_loss = computeLoss(y_val, val_preds)
        #calculate the numerical validation loss using the predictions and actual validation labels
        vLoss = computeLoss(tf.cast(y_val, tf.int32), val_preds).numpy()

        # Calculate validation accuracy
        #compare the highest probability prediction (argmax) to the actual label integer
        correct = tf.equal(tf.argmax(val_preds, axis=1, output_type=tf.int32), tf.cast(y_val, tf.int32))
        #calculate the mean of the correct predictions to get the accuracy percentage
        v_acc = tf.reduce_mean(tf.cast(correct, tf.float32)).numpy()

        #append the results of the current epoch to our tracking dictionary
        history['val_loss'].append(vLoss)
        history['val_acc'].append(v_acc)

        #print the progress to the console every 2 epochs
        if (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch + 1}: Loss {vLoss:.4f}, Acc {v_acc:.4f}")

        # Early stopping logic
        #wait until there is  at least 20 epochs of history to evaluate
        if len(history['val_loss']) > 20:
            #grab the last 10 validation loss values
            recent = history['val_loss'][-10:]
            #if the current validation loss is nota s good as the mean + standard deviation of the previous 9
            # then we have the model is overfitting/diverging, so we stop training
            if vLoss > np.mean(recent[:-1]) + np.std(recent[:-1]):
                print(f"Early stop at epoch {epoch + 1}")
                #break out of the 100 epoch loop
                break

    #return the dictionary containing the loss and accuracy history
    return history
#PART 4
def train_NN_semi_supervised(X_labeled, y_labeled, X_unlabeled, X_val, y_val, lr, alpha, batch_size):
    #dictionary to store validation loss and accuracy over epochs
    history = {'val_loss': [], 'val_acc': []}
    #get the total number of available labeled and unlabeled samples
    numLabeled = X_labeled.shape[0]
    num_Unlabeled = X_unlabeled.shape[0]
    #set the learning rate that will be adjusted during training
    adaptive_lr = lr

    #loop through a maximum of 100 training cycles (epochs)
    for epoch in range(100):
        # Step Decay (from Stage 3)
        #reduce the learning rate by 10% after every 2 epochs
        if epoch > 0 and epoch % 2 == 0:
            adaptive_lr *= 0.9

        # Calculate a weight that starts at 0.0 at epoch 0 and linearly increases to 1.0 by epoch 100
        # This prevents the noisy unlabeled data from confusing the model early on
        lambdaWeight = tf.cast(epoch / 100.0, tf.float32)

        # Shuffle indices at start of epoch for the labeled data
        indices = np.random.permutation(numLabeled)
        X_sh = X_labeled[indices]
        y_sh = y_labeled[indices]

        #loop through the labeled data in chunks of batch_size
        for i in range(0, numLabeled, batch_size):
            # A. Get Supervised Labeled batch
            xLabeledBatch = tf.cast(X_sh[i:i + batch_size], tf.float32)
            yLabeledBatch = tf.cast(y_sh[i:i + batch_size], tf.int32)

            # B. Get Unlabeled Batch and augment it
            # Randomly pick index numbers from the large pool of unlabeled data
            unlabeled_idx = np.random.choice(num_Unlabeled, batch_size)
            #grab the actual clean unlabeled images using those random indexes
            x_unlabeled_clean = X_unlabeled[unlabeled_idx]

            # Create a noisy/flipped physical copy of those same unlabeled images
            x_unlabeled_aug = flip_and_noise(x_unlabeled_clean)

            # Cast both the clean and augmented unlabeled data into float32 for TensorFlow
            x_unlabeled_clean = tf.cast(x_unlabeled_clean, tf.float32)
            x_unlabeled_aug = tf.cast(x_unlabeled_aug, tf.float32)

            #open a GradientTape to track operations for backpropagation
            with tf.GradientTape() as tape:
                # C. Supervised Forward Pass & Loss
                #do a forward pass on the labeled data (using the optimal 0.2 dropout rate)
                preds_labeled = modelForward(xLabeledBatch, training=True, dropout_rate=0.2)
                #calculate the standard cross-entropy loss using the known labels
                supervised_loss = computeLoss(yLabeledBatch, preds_labeled)

                # D. Semi-Supervised Forward Passes
                #get the network's raw guesses for both the clean and the noisy unlabeled images
                preds_clean = modelForward(x_unlabeled_clean, training=True, dropout_rate=0.2)
                preds_aug = modelForward(x_unlabeled_aug, training=True, dropout_rate=0.2)

                # Convert the raw logits from both passes into probability distributions (percentages)
                probs_clean = tf.nn.softmax(preds_clean)
                probs_aug = tf.nn.softmax(preds_aug)

                # Freeze the clean prediction so it becomes the 'correct' ground-truth target
                target_probs = tf.stop_gradient(probs_clean)

                # Calculate Mean Squared Error between the frozen clean target and the noisy prediction
                consistency_loss = tf.reduce_mean(
                    tf.reduce_sum(tf.square(target_probs - probs_aug), axis=1)
                )

                # Combine the supervised loss with the consistency loss
                # multiply the consistency loss by lambda so it slowly turns on over the 100 epochs
                total_loss = supervised_loss + (lambdaWeight * consistency_loss)

            # Perform backpropagation on the total combined loss
            #calculate the gradients of the total loss with respect to all network variables (weights/biases)
            grads = tape.gradient(total_loss, variables)

            # Manual StochasticGD update with momentum
            #loop through every variable (W1, b1, W2, b2, W3, b3)
            for j in range(len(variables)):
                #calculate the new momentum/inertia based on the alpha factor and the current gradient
                inertia[j] = alpha * inertia[j] + grads[j]
                #update the weight/bias variable by subtracting the adjusted learning step
                variables[j].assign_sub(adaptive_lr * inertia[j])

        # Validation evaluation
        #do a forward pass on the entire validation dataset (training=False turns off dropout)
        val_preds = modelForward(tf.cast(X_val, tf.float32), training=False)
        #calculate the numerical validation loss using the predictions and actual validation labels
        v_loss = computeLoss(tf.cast(y_val, tf.int32), val_preds).numpy()

        # Calculate validation accuracy
        #compare the highest probability prediction (argmax) to the actual label integer
        correct = tf.equal(tf.argmax(val_preds, axis=1, output_type=tf.int32), tf.cast(y_val, tf.int32))
        #calculate the mean of the correct predictions to get the accuracy percentage
        v_acc = tf.reduce_mean(tf.cast(correct, tf.float32)).numpy()

        #append the results of the current epoch to our tracking dictionary
        history['val_loss'].append(v_loss)
        history['val_acc'].append(v_acc)

        #print the progress to the console every 2 epochs, including the current lambda weight
        if (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch + 1}: Val Loss {v_loss:.4f}, Val Acc {v_acc:.4f} | Lambda: {lambdaWeight:.2f}")

        # Early stopping logic (with patience = 20)
        #wait until we have at least 20 epochs of history to evaluate
        if len(history['val_loss']) > 20:
            #grab the last 10 validation loss values
            recent = history['val_loss'][-10:]
            #if the current validation loss is worse than the mean + standard deviation of the previous 9
            # it means the model is overfitting/diverging, so we stop training early
            if v_loss > np.mean(recent[:-1]) + np.std(recent[:-1]):
                print(f"Early stop at epoch {epoch + 1}")
                #break out of the 100 epoch loop
                break

    #return the dictionary containing the loss and accuracy history
    return history



def evaluateTestSet(X_test, y_test):
    # Perform a single forward pass without dropout
    testLogits = modelForward(tf.cast(X_test, tf.float32), training=False)

    # Calculate accuracy
    correct = tf.equal(tf.argmax(testLogits, axis=1, output_type=tf.int32), tf.cast(y_test, tf.int32))
    testAcc = tf.reduce_mean(tf.cast(correct, tf.float32)).numpy()

    return testAcc

# PARTs 2&3 use this

"""history = train_NN(X_train_final, y_train_final, val_X, val_y,
                   lr=0.2, alpha=0.9, batch_size=128)

# Performance plotting
print('seed #1')
plt.plot(history['val_acc'], label='Validation Accuracy')
plt.title(' Performance for seed #1 dropout_rate=0.5')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend()
plt.show()
"""

#PART 4 uses this

# 1. Run Supervised Baseline
print("--- Training Supervised Model ---")
np.random.seed(42)
tf.random.set_seed(42)
resetWeights() # Ensuring random starting weights
history_sup = trainNN(trainDataFinalX, trainDataFinalY, val_X, val_y,
                      lr=0.2, alpha=0.9, batch_size=128)

# 2. Run Semi-Supervised Model
print("\n--- Training Semi-Supervised Model ---")
np.random.seed(42)
tf.random.set_seed(42)
resetWeights()
history_semi = train_NN_semi_supervised(trainDataFinalX, trainDataFinalY, unlabeledX, val_X, val_y,
                                        lr=0.2, alpha=0.9, batch_size=128)

# 3. Plot Both on One Graph
plt.figure(figsize=(10, 6))
plt.plot(history_sup['val_acc'], label='Supervised Baseline', linestyle='--')
plt.plot(history_semi['val_acc'], label='Semi-Supervised (Method C)')

plt.title('Validation Accuracy: Supervised vs. Semi-Supervised (Seed #42)')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend()
plt.show()

final_test_acc = evaluateTestSet(testingDataX, testingDataY)
print(f"\n>>> FINAL TEST ACCURACY: {final_test_acc:.4f} <<<")