from pyclustering.utils import linear_sum, square_sum

from pyclustering.cluster.encoder import type_encoding

from pyclustering.container.cftree import cftree, cfentry, measurement_type

import numpy as np

from mpl_toolkits.mplot3d import Axes3D

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import time

class birch:
    def __init__(self, data, number_clusters, branching_factor=5, max_node_entries=5, initial_diameter=0.1,
                type_measurement=measurement_type.CENTROID_EUCLIDEAN_DISTANCE,
                entry_size_limit=500,
                diameter_multiplier=1.5,
                ccore=True):
        self.__pointer_data = data
        self.__number_clusters = number_clusters
        
        self.__measurement_type = type_measurement
        self.__entry_size_limit = entry_size_limit
        self.__diameter_multiplier = diameter_multiplier
        self.__ccore = ccore

        self.__verify_arguments()

        self.__features = None
        self.__tree = cftree(branching_factor, max_node_entries, initial_diameter, type_measurement)
        
        self.__clusters = []
        self.__noise = []


    def process(self):
        self.__insert_data()
        self.__extract_features()

        current_number_clusters = len(self.__features)
        
        while current_number_clusters > self.__number_clusters:
            indexes = self.__find_nearest_cluster_features()
            
            self.__features[indexes[0]] += self.__features[indexes[1]]
            self.__features.pop(indexes[1])
            
            current_number_clusters = len(self.__features)
            
        # decode data
        self.__decode_data()
        return self
    
    
    def get_clusters(self):
        return self.__clusters


    def get_cluster_encoding(self):    
        return type_encoding.CLUSTER_INDEX_LIST_SEPARATION


    def __verify_arguments(self):
        if len(self.__pointer_data) == 0:
            raise ValueError("Input data is empty (size: '%d')." % len(self.__pointer_data))

        if self.__number_clusters <= 0:
            raise ValueError("Amount of cluster (current value: '%d') for allocation should be greater than 0." %
                            self.__number_clusters)

        if self.__entry_size_limit <= 0:
            raise ValueError("Limit entry size (current value: '%d') should be greater than 0." %
                            self.__entry_size_limit)


    def __extract_features(self):
        self.__features = []
        
        if len(self.__tree.leafes) == 1:
            # parameters are too general, copy all entries
            for entry in self.__tree.leafes[0].entries:
                self.__features.append(entry)

        else:
            # copy all leaf clustering features
            for node in self.__tree.leafes:
                self.__features.append(node.feature)
    
    
    def __decode_data(self):
        self.__clusters = [[] for _ in range(self.__number_clusters)]
        self.__noise = []
        
        for index_point in range(0, len(self.__pointer_data)):
            (_, cluster_index) = self.__get_nearest_feature(self.__pointer_data[index_point], self.__features)
            
            self.__clusters[cluster_index].append(index_point)
    
    
    def __insert_data(self):
        for index_point in range(0, len(self.__pointer_data)):
            point = self.__pointer_data[index_point]
            self.__tree.insert_cluster([point])
            
            if self.__tree.amount_entries > self.__entry_size_limit:
                self.__tree = self.__rebuild_tree(index_point)
        
        #self.__tree.show_feature_destibution(self.__pointer_data);
    
    
    def __rebuild_tree(self, index_point):
        rebuild_result = False
        increased_diameter = self.__tree.threshold * self.__diameter_multiplier
        
        tree = None
        
        while rebuild_result is False:
            # increase diameter and rebuild tree
            if increased_diameter == 0.0:
                increased_diameter = 1.0
            
            # build tree with update parameters
            tree = cftree(self.__tree.branch_factor, self.__tree.max_entries, increased_diameter, self.__tree.type_measurement)
            
            for index_point in range(0, index_point + 1):
                point = self.__pointer_data[index_point]
                tree.insert_cluster([point])
            
                if tree.amount_entries > self.__entry_size_limit:
                    increased_diameter *= self.__diameter_multiplier
                    continue
            
            # Re-build is successful.
            rebuild_result = True
        
        return tree
    
    
    def __find_nearest_cluster_features(self):
        minimum_distance = float("Inf")
        index1 = 0
        index2 = 0
        
        for index_candidate1 in range(0, len(self.__features)):
            feature1 = self.__features[index_candidate1]
            for index_candidate2 in range(index_candidate1 + 1, len(self.__features)):
                feature2 = self.__features[index_candidate2]
                
                distance = feature1.get_distance(feature2, self.__measurement_type)
                if distance < minimum_distance:
                    minimum_distance = distance
                    
                    index1 = index_candidate1
                    index2 = index_candidate2
        
        return [index1, index2]
    
    
    def __get_nearest_feature(self, point, feature_collection):
        minimum_distance = float("Inf")
        index_nearest_feature = -1
        
        for index_entry in range(0, len(feature_collection)):
            point_entry = cfentry(1, linear_sum([ point ]), square_sum([point]))
            
            distance = feature_collection[index_entry].get_distance(point_entry, self.__measurement_type)
            if distance < minimum_distance:
                minimum_distance = distance
                index_nearest_feature = index_entry
                
        return minimum_distance, index_nearest_feature


if __name__ == '__main__':
    datas = np.random.rand(500, 3)*100
    a = datas.tolist()

    start_time = time.time()
    b = birch(a, 8)
    b.process()
    run_time = time.time() - start_time

    colors = ['ro', 'bo', 'go', 'yo', 'r^', 'b^', 'g^', 'y^']
    g = []

    for i in range(len(b._birch__clusters)):
        g.append(datas[b._birch__clusters[i]])

    # for i in range(len(g)):
    #     plt.plot(g[i].T[0],g[i].T[1], colors[i])


    fig = plt.figure()
    graph = Axes3D(fig)
    for i in range(len(g)):
        graph.plot(g[i].T[0],g[i].T[1], g[i].T[2], colors[i])

    plt.show()
    print("Excute algorithm in %s seconds", run_time)