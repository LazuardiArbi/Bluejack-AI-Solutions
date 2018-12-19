import numpy as np
import tensorflow as tf
import csv
import random
import sys
from sklearn.decomposition import PCA

def load_data(path):
    dataset = []
    with open(path) as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            feature1, feature2, feature3, feature4, feature5 = row[5:11], row[12:13], row[14:16], row[17:21], row[23:24]
            feature = feature1 + feature2 + feature3 + feature4 + feature5
            feature = [float(x) for x in feature]
            result = row[4]
            dataset.append((feature, result))
    return dataset

def preprocessing(dataset):
    preprocess = []
    result_list = ['a','b','c','d','e']
    max_list = [sys.float_info.min for x in range(len(dataset))]
    min_list = [sys.float_info.max for x in range(len(dataset))]

    for features, result in dataset:
        for x in range(len(features)):
            if features[x] > max_list[x]:
                max_list[x] = features[x]
            if features[x] < min_list[x]:
                min_list[x] = features[x]

    for features, result in dataset:
        for x in range(len(features)):
            features[x] = normalize(features[x], min_list[x], max_list[x])

        result_index = result_list.index(result)
        new_result = np.zeros(len(result_list),'int')
        new_result[result_index] = 1

        preprocess.append((features,new_result))
    return preprocess

def split_data(dataset):
    feature_data = []
    result_data = []
    for feature, result in dataset:
        feature_data.append(feature)
        result_data.append(result)
    return feature_data, result_data

def normalize(data, min, max):
    normalized = (data - min) / (max - min) 
    return normalized

def applyPCA(data):
    pca = PCA(n_components=5)
    dataset_pca = pca.fit_transform(data)
    return dataset_pca

def collapse_data(feature_dataset, result_dataset, length):
    dataset_collapse = []
    for x in range(length):
        dataset_collapse.append((feature_dataset[x], result_dataset[x]))
    return dataset_collapse

dataset = load_data("O192-COMP7117-AS01-00-classification.csv")
dataset = preprocessing(dataset)

feature_dataset, result_dataset = split_data(dataset)
feature_dataset = applyPCA(feature_dataset)
dataset = collapse_data(feature_dataset, result_dataset, len(dataset))

random.shuffle(dataset)

counterTraining = int(0.7 * len(dataset))
counterValidation = int(0.2 * len(dataset))
training_dataset = dataset[:counterTraining]
validation_dataset = dataset[counterTraining:(counterTraining + counterValidation)]
test_dataset = dataset[(counterTraining + counterValidation):]

def full_connected(input, numinput, numoutput):
    w = tf.Variable(tf.random_normal([numinput,numoutput]))
    b = tf.Variable(tf.random_normal([numoutput]))
    wx_b = tf.matmul(input,w)+b
    act = tf.nn.sigmoid(wx_b)
    return act

num_of_input = 5
num_of_output = 5
num_of_hidden = [5,4]

def build_model(input):
    layer1 = full_connected(input,num_of_input,num_of_hidden[0])
    layer2 = full_connected(layer1,num_of_hidden[0],num_of_hidden[1])
    layer3 = full_connected(layer2,num_of_hidden[1],num_of_output)
    return layer3

trn_input = tf.placeholder(tf.float32,[None, num_of_input])
trn_target = tf.placeholder(tf.float32,[None, num_of_output])

learning_rate = .5
num_of_epoch = 5000
report_between = 500

def optimize(model, training_dataset, validation_dataset, test_dataset):
    error = tf.reduce_mean(.5 * (trn_target-model)**2)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(error)
    correct_prediction = tf.equal(tf.argmax(model,1), tf.argmax(trn_target,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(num_of_epoch + 1):
            feature_training = [data[0] for data in training_dataset]
            result_training = [data[1] for data in training_dataset]
            feed_training = {trn_input: feature_training, trn_target: result_training}
            _, error_value_training, accuracy_value_training = sess.run([optimizer, error, accuracy], feed_training)

            
            if epoch % report_between == 0:
                print("| Training Dataset \t-\t |Epoch : {0:6d} |\t Error : {1:12f} |".format(epoch, error_value_training*100))

            if epoch >= 500:
                feature_validation = [data[0] for data in validation_dataset]
                result_validation = [data[1] for data in validation_dataset]
                feed_validation = {trn_input: feature_validation, trn_target: result_validation}
                _, error_value_validation, accuracy_value_validation = sess.run([optimizer, error, accuracy], feed_validation)

                if epoch % report_between == 0:
                    print("| Validation Dataset \t-\t |Epoch : {0:6d} |\t Error : {1:12f} |".format(epoch, error_value_validation*100))
            
            if epoch % report_between == 0:
                print()

        feature_test = [data[0] for data in test_dataset]
        result_test = [data[1] for data in test_dataset]
        feed_test = {trn_input: feature_test, trn_target: result_test}
        error_value_test, accuracy_value_test = sess.run([error ,accuracy], feed_test)

        print("Tasting Statistic \t-\t Accuracy: ", accuracy_value_test*100)
                
model = build_model(trn_input)
optimize(model, training_dataset, validation_dataset, test_dataset)