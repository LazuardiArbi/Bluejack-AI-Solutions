import tensorflow as tf
import numpy as np
import csv
import random
import sys
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

class SOM:
    def __init__(self, height, width, inp_dimension):
        # step 1 inisialisasi
        self.height = height
        self.width = width
        self.inp_dimension = inp_dimension

        # step 2 inisialisasi node
        self.node = [tf.to_float([i,j]) for i in range(height) for j in range(width)]
        self.input = tf.placeholder(tf.float32,[inp_dimension])
        self.weight = tf.Variable(tf.random_normal([height*width, inp_dimension]))

        # buat fungsi cari node terdekat, update weight
        self.best_matching_unit = self.get_bmu(self.input)
        self.updated_weight = self.get_update_weight(self.best_matching_unit, self.input)
    
    def get_bmu(self, input):
        # hitung distances terpendek pake euclidean
        expand_input = tf.expand_dims(input, 0)
        euclidean = tf.square(tf.subtract(expand_input, self.weight))
        distances = tf.reduce_sum(euclidean,1) # 0 akan menambahkan secara kesamping

        # cari index distance terpendek
        min_index = tf.argmin(distances, 0) # 1 akan mengkecek ke samping | 0 ke bawah
        bmu_location = tf.stack([tf.mod(min_index, self.width), tf.div(min_index, self.width)])
        return tf.to_float(bmu_location)

    def get_update_weight(self, bmu, input):
        # inisialisasi learning rate & sigma
        learning_rate = 0.5
        sigma = tf.to_float(tf.maximum(self.height, self.width)/ 2)

        # hitung perbedaan jarak bmu ke node lain pake euclidean(bmu = best matching unit)
        expand_bmu = tf.expand_dims(bmu, 0)
        euclidean = tf.square(tf.subtract(expand_bmu, self.node))
        distances = tf.reduce_sum(euclidean, 1)
        ns = tf.exp(tf.negative(tf.div(distances**2, 2*sigma**2)))

        # hitung rate
        rate = tf.multiply(ns, learning_rate)
        num_of_node = self.height * self.width
        rate_value = tf.stack([tf.tile(tf.slice(rate,[i],[1]), [self.inp_dimension]) for i in range(num_of_node)])

        # hitung update weight-nya
        weight_diff = tf.multiply(rate_value, tf.subtract(tf.stack([input for i in range (num_of_node)]), self.weight))

        updated_weight = tf.add(self.weight, weight_diff)
        return tf.assign(self.weight, updated_weight)
    
    def train(self, dataset, num_of_epoch):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range (num_of_epoch):
                for data in dataset:
                    feed = { self.input : data}
                    sess.run([self.updated_weight],feed)
            self.weight_value  = sess.run(self.weight)
            self.node_value = sess.run(self.node)
            cluster = [ [] for i in range(self.width)]
            for i, location in enumerate(self.node_value): # diakan meng-return index-nya
                cluster[int(location[0])].append(self.weight_value[i])
            self.cluster = cluster

def load_data(path):
    dataset = []
    with open(path) as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            feature2 = row[3:5] # energy, energy-from-fat
            feature3 = row[6:9] # saturated-fat, monounsaturated-fat, polyunsaturated-fat
            feature4 = row[10:11] # trans-fat 
            feature5 = row[12:13] # sugars
            feature6 = row[14:16] # proteins, salt
            feature = feature2 + feature3 + feature4 + feature5 + feature6
            feature = [float(x) for x in feature]
            dataset.append(feature)
    return dataset

def feature_selection(dataset):
    feature_selection_dataset = []
    for feature in dataset:
        # caloprie density calcualtion
        calorie_density = (feature[0] + feature[1]) / 100
        
        # fat ratio calculation 
        dividend = (feature[3] + feature[4])
        devisor = (feature[2] + feature[5])
        if devisor == 0:
            devisor = 1
        fat_ratio = dividend / devisor
        feature_selection_dataset.append((calorie_density, fat_ratio, feature[6],feature[7],feature[8]))
    return feature_selection_dataset

def normalizeing(dataset):
    preprocess = []
    max_list = [sys.float_info.min for x in range(5)]
    min_list = [sys.float_info.max for x in range(5)]

    for features in dataset:
        for x in range(len(features)):
            if features[x] > max_list[x]:
                max_list[x] = features[x]
            if features[x] < min_list[x]:
                min_list[x] = features[x]

    for features in dataset:
        features = [float(x) for x in features]
        for x in range(len(features)):
            features[x] = normalize(features[x], min_list[x], max_list[x])
        preprocess.append(features)
        
    return preprocess

def normalize(data, min, max):
    normalized = (data - min) / (max - min) 
    return normalized

def applyPCA(data):
    pca = PCA(n_components=3)
    dataset_pca = pca.fit_transform(data)
    return dataset_pca

dataset = load_data("O192-COMP7117-AS01-00-clustering.csv")
dataset = feature_selection(dataset)
dataset = normalizeing(dataset)
dataset = applyPCA(dataset)
random.shuffle(dataset)
dataset = np.array(dataset)

som = SOM(5,5,3)
som.train(dataset, 5000)

plt.imshow(som.cluster)
plt.show()