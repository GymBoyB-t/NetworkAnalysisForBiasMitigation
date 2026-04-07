import numpy as np
import umap.umap_ as umap
import subsets
import datasets
import datasetProcessing
import graph
import subsets_percentageBased

image_save_path = r"C:\Users\user\Desktop\DSCI_Final_Project\TestImages"
subset_save_path = r"C:\Users\user\Desktop\DSCI_Final_Project\subsets"

images, labels = datasets.loadMNIST()

#-------------------------------------------
#-------------------------------------------
####### PARAMETERS ########

#If you want the training dataset to be unbiased 
# (false means all classes have equal sample count, 
# true means classes are biased, use bias array to change class weights)
unbalanced = True
bias_array = [0.001, 0.001, 0.001, 0.01, 0.01, 0.01, 0.05, 0.05, 0.05, 0.05]

# This is entirely redacted I just don't have time to remove it, does nothing
total_subset_size = 500

# This is the minimum number of samples for each class, if a class has less 
# than this number of sample it won't be subsetted
samples_per_class = 500

#-------------------------------------------
#-------------------------------------------

#Distribution names were used debuging only, ignore this
unbalancedDistribution = "unbalancedDistributionType1"

#This is a string that is added to file names so you can tell what experiment the file is from
settingsText = f";subsetSize_{total_subset_size};samples_per_class_{samples_per_class};isUnbalanced;{unbalanced};{unbalancedDistribution}"

#Auto generate paths
train_subset_path = f"train_full;samples_per_class_{samples_per_class};isUnbalanced;{unbalanced};{unbalancedDistribution}"
test_subset_path = f"test_full;samples_per_class_{samples_per_class};isUnbalanced;{unbalanced};{unbalancedDistribution}"


if __name__ == "__main__":

    if unbalanced:
        X_train, X_test, y_train, y_test = datasetProcessing.get_unbalancedClassSubset(
            images, labels,
            bias_array
        )
    else:
        
        X, y = datasetProcessing.get_classSubset(images, labels, samples_per_class)

    unique, counts = np.unique(y_train, return_counts=True)
    total = len(y_train)

    print("Class distribution training set:")
    for cls, count in zip(unique, counts):
        print(f"Class {cls}: {count} samples ({count/total:.4f})")

    unique, counts = np.unique(y_test, return_counts=True)
    total = len(y_test)

    print("Class distribution test set:")
    for cls, count in zip(unique, counts):
        print(f"Class {cls}: {count} samples ({count/total:.4f})")

    graph.normalizeXforSimilarityCalculations(X_train)

    print("Building index...")
    index = graph.get_similaritySearchIndex(X_train)

    print("Building graph...")
    G = graph.build_knn_graph(X_train, index)

    print("Building subsets...")
    subsets.save_subset(train_subset_path, np.arange(len(X_train)) ,X_train ,y_train ,subset_save_path)
    subsets.save_subset(test_subset_path, np.arange(len(X_test)),X_test,y_test,subset_save_path)


    subsets_percentageBased.class_network_filter_subset(X_train, y_train, G, total_subset_size, samples_per_class, subset_save_path, settingsText)
    subsets_percentageBased.high_degree_subset(X_train, y_train, G, total_subset_size, samples_per_class, subset_save_path, settingsText)
    
    subsets_percentageBased.high_clustering_subset(X_train, y_train, G, total_subset_size, samples_per_class, subset_save_path, settingsText)
    subsets_percentageBased.low_degree_subset(X_train, y_train, G, total_subset_size, samples_per_class, subset_save_path, settingsText)

    subsets_percentageBased.random_subset(
        X_train, y_train, total_subset_size, samples_per_class, subset_save_path, settingsText
    )

    subsets_percentageBased.high_degree_subset(
        X_train, y_train, G, total_subset_size, samples_per_class, subset_save_path, settingsText
    )

    subsets_percentageBased.low_degree_subset(
        X_train, y_train, G, total_subset_size, samples_per_class, subset_save_path, settingsText
    )

    subsets_percentageBased.high_clustering_subset(
        X_train, y_train, G, total_subset_size, samples_per_class, subset_save_path, settingsText
    )

    subsets_percentageBased.low_clustering_subset(
        X_train, y_train, G, total_subset_size, samples_per_class, subset_save_path, settingsText
    )

    subsets_percentageBased.high_pathlength_subset(
        X_train, y_train, G, total_subset_size, samples_per_class, subset_save_path, settingsText
    )

    subsets_percentageBased.low_pathlength_subset(
        X_train, y_train, G, total_subset_size, samples_per_class, subset_save_path, settingsText
    )

    subsets_percentageBased.high_diameter_subset(
        X_train, y_train, G, total_subset_size, samples_per_class, subset_save_path, settingsText
    )

    subsets_percentageBased.low_diameter_subset(
        X_train, y_train, G, total_subset_size, samples_per_class, subset_save_path, settingsText
    )

   # subsets_percentageBased.high_density_subset(
   #     X_train, y_train, G, total_subset_size, samples_per_class, subset_save_path, settingsText
   # )

   # subsets_percentageBased.low_density_subset(
   #     X_train, y_train, G, total_subset_size, samples_per_class, subset_save_path, settingsText
   # )

"""
    print("Creating abnormal subsets")
    subsets.random_subset(X_train, y_train, total_subset_size, samples_per_class, subset_save_path, settingsText)

    subsets.class_network_filter_subset(X_train, y_train, G, total_subset_size, samples_per_class, subset_save_path, settingsText)
    ##########

    subsets.high_degree_subset(X_train, y_train, G, total_subset_size, samples_per_class, subset_save_path, settingsText)
    subsets.low_degree_subset(X_train, y_train, G, total_subset_size, samples_per_class, subset_save_path, settingsText)

    subsets.high_clustering_subset(X_train, y_train, G, total_subset_size, samples_per_class, subset_save_path, settingsText)
    subsets.low_clustering_subset(X_train, y_train, G, total_subset_size, samples_per_class, subset_save_path, settingsText)

    subsets.high_pathlength_subset(X_train, y_train, G, total_subset_size, samples_per_class, subset_save_path, settingsText)
    subsets.low_pathlength_subset(X_train, y_train, G, total_subset_size, samples_per_class, subset_save_path, settingsText)

    subsets.high_diameter_subset(X_train, y_train, G, total_subset_size, samples_per_class, subset_save_path, settingsText)
    subsets.low_diameter_subset(X_train, y_train, G, total_subset_size, samples_per_class, subset_save_path, settingsText)

    subsets.high_density_subset(X_train, y_train, G, total_subset_size, samples_per_class, subset_save_path, settingsText)
    subsets.low_density_subset(X_train, y_train, G, total_subset_size, samples_per_class, subset_save_path, settingsText)

"""
    

