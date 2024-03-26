import numpy as np
import sklearn.cluster

class KNNClassifier:
    def __init__(self, k, x_train, y_train):
        self.k = k
        self.x_train = x_train
        self.y_train = y_train

    # given a query of gene data, returns a numpy array with prediction labels
    # for each sample
    def predict(self, x_query):
        y_pred = []

        for query in x_query:
            distances = []
            for train in self.x_train:
                distance = np.sqrt(np.sum(np.square(query - train)))
                distances.append(distance)
                
            k_closest_indices = np.argsort(distances)[:self.k]
            
            k_labels = [self.y_train[i] for i in k_closest_indices]
            y_pred.append(max(set(k_labels), key=k_labels.count))

        return np.array(y_pred)
    
    
#given a file name gets training data and labels
def parse_data(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        samples = lines[0].strip().split()
        n = len(samples)
        f = len(lines[1:len(lines)-1])
        x_train = np.empty((n, f))
        genes = []
        for i in range(1,1+f):
            gene = lines[i].strip().split()
            gene = gene[1:]
            genes.append(gene)

        for i in range(len(genes[0])):
            for j in range(len(genes)):
                x_train[i][j] = genes[j][i]
        
        last_line = lines[-1].strip()  
        classes = last_line.split()  
        
        y_train = np.array(classes[1:len(classes)])
        return x_train, y_train
    
# given a numpy array and filename prints the sample with the prediction file
def print_predictions_to_file(y_pred, filename):
    with open(filename, 'w') as file:
        for i in range(len(y_pred)):
            file.write("PATIENT" + str(i+31) + "\t" + y_pred[i])
            file.write("\n")

#returns the accuracy of prediction of y given true y labels
def accuracy(y_true, y_pred):
        correct = 0
        for i in range(len(y_pred)):
            if y_pred[i] == y_true[i]:
                correct += 1
        return correct / len(y_pred)

#given a dictionary and a filename, prints dictionary to the file
def print_dict_to_file(dict, filename):
    with open(filename, 'w') as file:
        for x in dict:
            file.write(str(x) + "\t".expandtabs(4) + str(dict[x]))
            file.write("\n")

# given a true labels and predicted labels, returns values for true positives
# true negatives, false positives, and false negatives
def calc_TP_TN_FP_FN(y_true, y_pred):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(y_true)):
        if y_true[i] == "CurrentSmoker" and y_pred[i] == "CurrentSmoker":
            TP+=1
        elif y_true[i] == "NeverSmoker" and y_pred[i] == "NeverSmoker":
            TN+=1
        elif y_true[i] == "NeverSmoker" and y_pred[i] == "CurrentSmoker":
            FP+=1
        else:
            FN+=1
    
    return TP, TN, FP, FN

#prints x and y values of given dict in separate files
def special_print(dict):
    with open("xvalues.txt", 'w') as file:
        for key in dict:
            file.write(str(key))
            file.write("\n")
    with open("yvalues.txt", 'w') as file:
        for key in dict:
            file.write(str(dict[key]))
            file.write("\n")

if __name__ == "__main__":

    x_train, y_train = parse_data("GSE994-train.txt")
    x_test, y_test = parse_data("GSE994-test.txt")

    knn1 = KNNClassifier(1, x_train, y_train)
    y_pred1 = knn1.predict(x_test)
    print_predictions_to_file(y_pred1, "Prob5-1NNoutput.txt")

    knn3 = KNNClassifier(3, x_train, y_train)
    y_pred3 = knn3.predict(x_test)
    print_predictions_to_file(y_pred3, "Prob5-3NNoutput.txt")

    folds = [{0, 1, 2, 3, 4, 5},
             {6, 7, 8, 9, 10, 11},
             {12, 13, 14, 15, 16, 17},
             {18, 19, 20, 21, 22, 23},
             {24, 25, 26, 27, 28, 29}]
    k_vals = [1, 3, 5, 7, 11, 21, 29]

    y_pred_by_k = {}
    acc_by_k = {}
    
    for k_val in k_vals:
        pred_for_k = np.array([])
        fold_accuracy_scores = []
        for fold in folds:
            x_fold = []
            y_fold = []
            x_excluded = []
            y_actual = []
            for i in range(30):
                if not (i in fold):
                    #print("taking i: " + str(i))
                    x_fold.append(x_train[i])
                    y_fold.append(y_train[i])
                else: # ommited fold
                    x_excluded.append(x_train[i])
                    y_actual.append(y_train[i])
            npxfold = np.array(x_fold)
            npyfold = np.array(y_fold)
            npxexcluded = np.array(x_excluded)
            cur_fold = KNNClassifier(k_val, npxfold, npyfold)
            y_fold_pred = cur_fold.predict(npxexcluded)
            pred_for_k = np.append(pred_for_k, y_fold_pred)
            acc = accuracy(y_actual, y_fold_pred)
            fold_accuracy_scores.append(acc)
        
        cur_acc = sum(fold_accuracy_scores) / len(fold_accuracy_scores)
        y_pred_by_k[k_val] = pred_for_k
        acc_by_k[k_val] = cur_acc

    print_dict_to_file(y_pred_by_k, "y_pred_by_k.txt")
    print_dict_to_file(acc_by_k, "acc_by_k.txt")

    special_print(acc_by_k)

    ac = sklearn.cluster.AgglomerativeClustering(n_clusters=2, 
                                                 linkage='average')
    
    ac.fit(x_train)
    labels = ac.labels_
    
    label_map = {0: "NeverSmoker", 1: "CurrentSmoker"}

    cluster_labels = np.array([label_map[label] for label in labels])

    TP1, TN1, FP1, FN1 = calc_TP_TN_FP_FN(y_train, cluster_labels)
    print("Cluster Model:")
    print(TP1)
    print(TN1)
    print(FP1)
    print(FN1)

    TP2, TN2, FP2, FN2 = calc_TP_TN_FP_FN(y_train, y_pred_by_k[5])
    print("KNN Classifier with k = 5")
    print(TP2)
    print(TN2)
    print(FP2)
    print(FN2)
