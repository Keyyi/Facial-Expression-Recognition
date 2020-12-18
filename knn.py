import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
    
def select_knn_model(improvement: bool):
    result = {}
    npzfile = np.load("./read_images/raf_db.npz")
    npzfile1 = np.load("./read_images/toronto_face.npz")
    
    train_images = npzfile["inputs_train"]
    train_labels = np.argmax(npzfile["target_train"], axis=1)
    test_images = npzfile["inputs_valid"]
    test_labels = np.argmax(npzfile["target_valid"], axis=1)
    
    train_images1 = npzfile1["inputs_train"]
    train_labels1 = npzfile1["target_train"]
    test_images1 = npzfile1["inputs_valid"]
    test_labels1 = npzfile1["target_valid"]
    
    for i in range(len(train_labels1)):
        if train_labels1[i] == 0:
            train_labels1[i] = 5
        elif train_labels1[i] == 1:
            train_labels1[i] = 2
        elif train_labels1[i] == 2:
            train_labels1[i] = 1
        elif train_labels1[i] == 3:
            train_labels1[i] = 3
        elif train_labels1[i] == 4:
            train_labels1[i] = 4
        elif train_labels1[i] == 5:
            train_labels1[i] = 0
        elif train_labels1[i] == 6:
            train_labels1[i] = 6
        else:
            print("wrong train label",train_labels1[i])
        
    for i in range(len(test_labels1)):
        if test_labels1[i] == 0:
            test_labels1[i] = 5
        elif test_labels1[i] == 1:
            test_labels1[i] = 2
        elif test_labels1[i] == 2:
            test_labels1[i] = 1
        elif test_labels1[i] == 3:
            test_labels1[i] = 3
        elif test_labels1[i] == 4:
            test_labels1[i] = 4
        elif test_labels1[i] == 5:
            test_labels1[i] = 0
        elif test_labels1[i] == 6:
            test_labels1[i] = 6
        else:
            print("wrong test label",test_labels1[i])
    #Get data
    X_train = np.concatenate((train_images, train_images1))
    X_valid = np.concatenate((test_images, test_images1))
    #Get label
    y_train = np.concatenate((train_labels, train_labels1))
    y_valid = np.concatenate((test_labels, test_labels1))
    
    best_k = -1
    best_perform = 0.0
    
    k_set = [(i+1) for i in range(20)]
    for value in k_set:
        print(value)
        if improvement:
            model = KNeighborsClassifier(n_neighbors = value,metric = 'cosine')
        else:
            model = KNeighborsClassifier(n_neighbors = value)
        model.fit(X_train,y_train.ravel())
        valid_predicted = model.predict(X_valid)
# Check validate set & get accuracy
        valid_count = 0
        for i in range(len(valid_predicted)):
            if valid_predicted[i] == y_valid[i]:
                valid_count += 1
                
        result[value] = valid_count/len(valid_predicted)
        
        print(result[value])
        
        if(result[value]>best_perform):
            best_k = value
            best_perform = result[value]
    print("Best performance K is",best_k)
    return result
    
def draw_graph(res,title):
    validation_set = [res[item] for item in range(1,21)]
    k_set = [(i+1) for i in range(20)]

    plt.plot(k_set,validation_set,label="validation")
    plt.xlabel("k")
    plt.ylabel("accuracy")
    plt.xticks(k_set)
    plt.title(title)
    plt.legend()
    plt.show()
    
###########------main------###########
if __name__ == '__main__':
    
    res = select_knn_model(False)
    draw_graph(res,"Original Graph")
    for i in range(1,5):
        print("When k is",i,", the validation accuracy is",res[i])
    
    improve_res = select_knn_model(True)
    draw_graph(improve_res,"Improvement Graph")
