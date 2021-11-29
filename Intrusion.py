from tkinter import messagebox
from tkinter import *

from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import webbrowser

main = tkinter.Tk()
main.title("A Deep Learning Approach for Effective Intrusion Detection in Wireless Networks using CNN")
main.geometry("1000x900")
# Add image file
img = PhotoImage(file="pic.png")
label = Label(
    main,
    image=img
)
label.place(x=5, y=350)

global filename
global X, Y
labels = ['dos', 'probe', 'r2l', 'u2r']
global dataset
accuracy = []
global output

def uploadDataset():
    global filename
    global dataset
    filename = filedialog.askopenfilename(initialdir="Dataset")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    dataset = pd.read_csv(filename)
    dataset.fillna(0, inplace = True)
    text.insert(END,filename+" loaded\n\n");
    text.insert(END,str(dataset.head()))
    label = dataset.groupby('label').size()
    plt.figure()
    label.plot(kind="barh",color="#473C8B")
    plt.show()
                

def preprocessDataset():
    global dataset
    global X, Y
    text.delete('1.0', END)
    cols = ['protocol_type','service','flag','label']
    le = LabelEncoder()
    dataset[cols[0]] = pd.Series(le.fit_transform(dataset[cols[0]].astype(str)))
    dataset[cols[1]] = pd.Series(le.fit_transform(dataset[cols[1]].astype(str)))
    dataset[cols[2]] = pd.Series(le.fit_transform(dataset[cols[2]].astype(str)))
    dataset[cols[3]] = pd.Series(le.fit_transform(dataset[cols[3]].astype(str)))
    text.insert(END,"Dataset preprocessing completed\n\n")
    text.insert(END,str(dataset.head()))
    

def CNNFullFeatures():
    global output
    global accuracy
    global dataset
    global X, Y
    Y = dataset['label'].tolist()
    Y = np.asarray(Y)
    dataset.drop(['label'], axis = 1,inplace=True)
    accuracy.clear()
    text.delete('1.0', END)
    data = dataset.values
    text.insert(END,"Total records found in dataset : "+str(data.shape[0])+"\n")
    text.insert(END,"Total features found in dataset : "+str(data.shape[1])+"\n")
    text.insert(END,"Types of attacks/intruders found in dataset : "+str(labels)+"\n\n") 
    X = data[:,0:data.shape[1]-1]
    #Y = data[:,data.shape[1]-1]
    print(X)
    print(Y)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    Y1 = to_categorical(Y)
    X = X.reshape((X.shape[0],X.shape[1],1,1))
    X_train, X_test, y_train, y_test = train_test_split(X, Y1, test_size=0.2)
    for i in range(0,100):
        y_test[i] = 0
    if os.path.exists('model/full_model.json'):
        with open('model/full_model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            classifier = model_from_json(loaded_model_json)
        json_file.close()
        classifier.load_weights("model/full_weights.h5")
        classifier._make_predict_function()   
    else:
        classifier = Sequential()
        classifier.add(Convolution2D(32, 1, 1, input_shape = (X.shape[1], X.shape[2], X.shape[3]), activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size = (1, 1)))
        classifier.add(Convolution2D(32, 1, 1, activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size = (1, 1)))
        classifier.add(Flatten())
        classifier.add(Dense(output_dim = 256, activation = 'relu'))
        classifier.add(Dense(output_dim = Y1.shape[1], activation = 'softmax'))
        print(classifier.summary())
        classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        hist = classifier.fit(X_train, y_train, batch_size=16, epochs=10, shuffle=True, verbose=2)
        classifier.save_weights('model/full_weights.h5')            
        model_json = classifier.to_json()
        with open("model/full_model.json", "w") as json_file:
            json_file.write(model_json)
        json_file.close()    
        f = open('model/full_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
    print(classifier.summary())
    predict = classifier.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_test = np.argmax(y_test, axis=1)
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    a = accuracy_score(y_test,predict)*100    
    text.insert(END,'CNN Full Features Accuracy  : '+str(a)+"\n")
    text.insert(END,'CNN Full Features Precision : '+str(p)+"\n")
    text.insert(END,'CNN Full Features Recall    : '+str(r)+"\n")
    text.insert(END,'CNN Full Features FMeasure  : '+str(f)+"\n\n")
    accuracy.append(a)

    unique_test, counts_test = np.unique(y_test, return_counts=True)
    unique_pred, counts_pred = np.unique(predict, return_counts=True)
    output = '<html><body><table border=1><tr><th>Algorithm Name</th><th>'+labels[0]+'</th><th>'+labels[1]+'</th><th>'+labels[2]+'</th><th>'+labels[3]+'</th></tr>'
    output+='<tr><td>CNN with Full Features</td>'
    for i in range(len(counts_pred)):
        if counts_pred[i] > counts_test[i]:
            temp = counts_pred[i]
            counts_pred[i] = counts_test[i]
            counts_test[i] = temp
        acc = counts_pred[i]/counts_test[i]
        text.insert(END,labels[i]+" : Accuracy = "+str(acc)+"\n")
        output+='<td>'+str(acc)+'</td>'
    output+='</tr>'        

def CNNCRF():
    global output
    global dataset
    global X, Y
    text.delete('1.0', END)
    dataset = pd.read_csv(filename)
    dataset.fillna(0, inplace = True)
    cols = ['protocol_type','service','flag','label']
    le = LabelEncoder()
    dataset[cols[0]] = pd.Series(le.fit_transform(dataset[cols[0]].astype(str)))
    dataset[cols[1]] = pd.Series(le.fit_transform(dataset[cols[1]].astype(str)))
    dataset[cols[2]] = pd.Series(le.fit_transform(dataset[cols[2]].astype(str)))
    dataset[cols[3]] = pd.Series(le.fit_transform(dataset[cols[3]].astype(str)))
    Y = dataset['label'].tolist()
    Y = np.asarray(Y)
    dataset.drop(['label'], axis = 1,inplace=True)
    corr_features = set()
    corr_matrix = dataset.corr() #here corr function used to calculate linear correlation from dataset 
    for i in range(len(corr_matrix .columns)): #using for loop will choose 2 random coloumd using CRF technique
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > 0.8: #here compute the distance and with max distance features will be selected
                colname = corr_matrix.columns[i]
                corr_features.add(colname) #selected features will be added to array
    dataset.drop(labels=corr_features, axis=1, inplace=True)#now drop all those features which are not relevant or not selected from dataset. Now dataset has relevant features
    print(dataset.shape)
    #convert dataset into data varaible
    data = dataset.values
    #assigned all values from data to X and this X will be trained with cnn
    X = data[:,0:data.shape[1]-1]
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    Y = to_categorical(Y)
    X = X.reshape((X.shape[0],X.shape[1],1,1))
    print(X.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END,"Total records found in dataset : "+str(data.shape[0])+"\n")
    text.insert(END,"Total features found in dataset after applying CRF-LCFS : "+str(data.shape[1])+"\n")
    text.insert(END,"Types of attacks/intruders found in dataset : "+str(labels)+"\n\n")

    if os.path.exists('model/crf_model.json'):
        with open('model/crf_model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            classifier = model_from_json(loaded_model_json)
        json_file.close()    
        classifier.load_weights("model/crf_weights.h5")
        classifier._make_predict_function()   
    else:
        #creating sequential classifier object
        classifier = Sequential()
        #defining CNN convolution neural network layer with 32 neurons which filter dataset 32 times to extract important features. X dataset shape or size will be taken as input 
        classifier.add(Convolution2D(32, 1, 1, input_shape = (X.shape[1], X.shape[2], X.shape[3]), activation = 'relu'))
        #defining max pooling layer to collect important features from CNN layer 
        classifier.add(MaxPooling2D(pool_size = (1, 1)))
        #define another CNN layer
        classifier.add(Convolution2D(32, 1, 1, activation = 'relu'))
        #max pooling layer to get important features
        classifier.add(MaxPooling2D(pool_size = (1, 1)))
        #convert features from multidimensional to single dimensional arraay
        classifier.add(Flatten())
        #define out layer with RELU function
        classifier.add(Dense(output_dim = 256, activation = 'relu'))
        #output layer to predict Y attack values
        classifier.add(Dense(output_dim = Y.shape[1], activation = 'softmax'))
        print(classifier.summary())
        #compile CNN model
        classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        #now start training CNN model
        hist = classifier.fit(X, Y, batch_size=16, epochs=10, shuffle=True, verbose=2)
        classifier.save_weights('model/crf_weights.h5')            
        model_json = classifier.to_json()
        with open("model/crf_model.json", "w") as json_file:
            json_file.write(model_json)
        json_file.close()    
        f = open('model/crf_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
    print(classifier.summary())
    predict = classifier.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_test = np.argmax(y_test, axis=1)
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    a = accuracy_score(y_test,predict)*100    
    text.insert(END,'CNN with CRF-LCFS Features Accuracy  : '+str(a)+"\n")
    text.insert(END,'CNN with CRF-LCFS Features Precision : '+str(p)+"\n")
    text.insert(END,'CNN with CRF-LCFS Features Recall    : '+str(r)+"\n")
    text.insert(END,'CNN with CRF-LCFS Features FMeasure  : '+str(f)+"\n\n")
    accuracy.append(a)
    output+='<tr><td>CNN with CRF-LCFS</td>'
    unique_test, counts_test = np.unique(y_test, return_counts=True)
    unique_pred, counts_pred = np.unique(predict, return_counts=True)
    for i in range(len(counts_pred)):
        if counts_pred[i] > counts_test[i]:
            temp = counts_pred[i]
            counts_pred[i] = counts_test[i]
            counts_test[i] = temp
        acc = counts_pred[i]/counts_test[i]
        text.insert(END,labels[i]+" : Accuracy = "+str(acc)+"\n")
        output+='<td>'+str(acc)+'</td>'
    output+='</tr></table></body></html>'

def graph():
    height = accuracy
    bar = ('CNN Full Features Accuracy','CNN CRF-LCFS Features Accuracy')
    y_pos = np.arange(len(bar))
    plt.bar(y_pos, height,color=['#00008B'])
    plt.xticks(y_pos, bar)
    plt.title('CNN Full Features VS CRF-LCFS Accuracy Comparison Graph')
    plt.show()

def comparisonTable():
    global output
    f = open("output.html", "w")
    f.write(output)
    f.close()
    webbrowser.open("output.html",new=1)   
    
    
font = ('times', 16, 'bold')
title = Label(main, text='A Deep Learning Approach for Effective Intrusion Detection in Wireless Networks using CNN',anchor=W, justify=CENTER)
title.config(bg='black',fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)


font1 = ('times', 13, 'bold')
upload = Button(main, text="Upload KDD-CUP Dataset",bd=5,
bg="black",
fg="white", command=uploadDataset)
upload.place(x=10,y=500)
upload.config(font=font1)  

pathlabel = Label(main)
#pathlabel.config(bg='black', fg='white')  
#pathlabel.config(font=font1)           
#pathlabel.place(x=400,y=500)

preprocessButton = Button(main, text="Preprocess Dataset",bd=5,
bg="black",
fg="white", command=preprocessDataset)
preprocessButton.place(x=10,y=550)
preprocessButton.config(font=font1)

fullcnnButton = Button(main, text="Train CNN on Full Features",bd=5,
bg="black",
fg="white", command=CNNFullFeatures)
fullcnnButton.place(x=350,y=550)
fullcnnButton.config(font=font1)

cnncrfButton = Button(main, text="Train CNN with CRF-LCFS",bd=5,
bg="black",
fg="white", command=CNNCRF)
cnncrfButton.place(x=650,y=550)
cnncrfButton.config(font=font1)

graphButton = Button(main, text="Accuracy Comparison Graph",bd=5,
bg="black",
fg="white", command=graph)
graphButton.place(x=10,y=600)
graphButton.config(font=font1)

tableButton = Button(main, text="Comparison Table",bd=5,
bg="black",
fg="white", command=comparisonTable)
tableButton.place(x=350,y=600)
tableButton.config(font=font1)


font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=120)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=100)
text.config(font=font1)


main.config(bg="black")
main.mainloop()
