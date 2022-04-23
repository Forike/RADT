#Data Introduction

The names of all subfolders under this folder represent the perturbation injection rate, and the files inside each subfolder differ only in the hdfs_test_***.txt file. This means that the model can be trained using the data in the "0.00" folder and evaluated on other datasets without retraining. 

All data has been processed according to Drain into the text file(https://github.com/logpai/logparser), where each line of the text file represents a sequence of log templates and the individual numbers represent the indexes of their templates. Otherwise, Hdfs_semantic_vec.json represents the mapping of each template index to the corresponding log semantic vector.

hdfs_train_1.txt and hdfs_train_2.txt correspond to D_1 and D_2 in the paper, respectively, and hdfs_test_abnormal.txt and hdfs_test_normal.txt are used to evaluate the model.
