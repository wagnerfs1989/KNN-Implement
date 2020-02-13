import pandas as pd
import random
import math
import operator
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict


def loadDataset(dataset,target, split):
    X_trainingSet = []
    X_testSet = []
    Y_trainingSet=[]
    Y_testSet=[]
#    with open(dataset, 'r') as csvfile:
#        lines = csv.reader(csvfile)
#        dataset = list(lines)
    for x in range(len(dataset)-1):
        if random.random() < split:
            X_trainingSet.append(dataset.loc[x])
            Y_trainingSet.append(target.loc[x])
        else:
            X_testSet.append(dataset.loc[x])
            Y_testSet.append(target.loc[x])
    return X_trainingSet,X_testSet,Y_trainingSet, Y_testSet
    
def euclideanDistance(instance1, instance2, lenght):
    distance = 0
    for x in range(lenght):
        distance+= pow((instance1[x] - instance2[x]),2)
    return math.sqrt(distance)

def getNeighbors(trainingSet, testInstance, k):
    distances = []
    lenght = len(testInstance)
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], lenght)
        distances.append((Y_train[x],dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x])
    return neighbors



def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][0]
        if response in classVotes:
            classVotes[response] += 1
        else:            
            classVotes[response] = 1
        sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]
    
def getAccuracy(testSet, predictions):
    correct = 0
    lenght = 0
    for x in range (len(testSet)):
        lenght+=1
        if Y_test[x] is predictions[x]:
            correct += 1
    return(correct/float(lenght)) * 100.0

def distribuicao (data):
    '''
    Esta função exibirá a quantidade de registros únicos para cada coluna
    existente no dataset
    
    dataframe -> Histogram
    '''
    # Calculando valores unicos para cada label: num_unique_labels
    num_unique_labels = data.apply(pd.Series.nunique)

    # plotando valores
    num_unique_labels.plot( kind='bar')
    
    # Nomeando os eixos
    plt.xlabel('Campos')
    plt.ylabel('Número de Registros únicos')
    plt.title('Distribuição de dados únicos do DataSet')
    
    # Exibindo gráfico
    plt.show()


def pie_chart(data,col1,col2,title): 
    labels = {'Comestivel':0,'Venenoso':1}
    sizes = data[col2]
    colors = ['#e5ffcc', '#ffb266']

    plt.pie(sizes, labels=labels, colors=colors,
                autopct='%1.1f%%', shadow=True, startangle=140, labeldistance =1.2)
    plt.title( title )
    
    plt.axis('equal')
    plt.show()


#def main():
    
fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)    
df = pd.read_csv(r'Dados\mushrooms.csv',sep = ',')
dataset = df
print(df.info())
print(df.describe())
distribuicao(df)

e = pd.value_counts(df['class']) [0]
p = pd.value_counts(df['class']) [1]

tam = len(df)

print('Cogumelos Comestiveis: ',e)
print('Cogumelos Venenosos: ',p )

pie = pd.DataFrame([['Comestivel',e],['Venenoso',p]],columns=['Tipo' , 'Quantidade'])

pie_chart(pie,'Tipo' , 'Quantidade','Distribuição Percentual Classes de Cogumelos')

plt.bar(pie.Tipo,pie.Quantidade, color = ['#e5ffcc', '#ffb266'])
plt.title("Distribuição das Classes de Cogumelos")
plt.xlabel("Tipo de Cogumelo")
plt.ylabel('Quantidade de Registros')
plt.show()
y = df['class']
df = df.drop('class', axis =1)
Oht_enc = OneHotEncoder()
df = pd.DataFrame(Oht_enc.fit_transform(df).A)
print(df.shape)
X_train,X_test,Y_train,Y_test = loadDataset(df,y, 0.3) 
scaler = StandardScaler()  
scaler.fit(X_train)

X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)

print('Train set: ' + repr(len(X_train)))
print('Test set: '+ repr(len(X_test)))
predictions = []
predictions_target = ['e','p']
k = 3
for x in range(len(X_test)):
    neighbors= getNeighbors(X_train, X_test[x], k)
    result = getResponse(neighbors)
    predictions.append(result)
#    print(' Predicted= ' + repr(result)+', actual=' +repr(Y_test[x]))
accuracy = getAccuracy(Y_test, predictions)
print('Accuracy: ' + repr(accuracy) + '%')    
# Imprimindo a matriz confusa
print("Matriz Confusa: ")
print(confusion_matrix(Y_test, predictions), "\n") 

# Imprimindo o relatório de classificação
print("Relatório de classificação: \n", classification_report(Y_test, predictions))  

# TESTE


X = []
for iclass in range(len(pie)):
    X.append([[], [], []])
    for i in range(len(predictions)):
        if predictions[i] == predictions_target[iclass]:
            X[iclass][0].append(X_test[i][0])
            X[iclass][1].append(X_test[i][1])
            X[iclass][2].append(X_test[i][2])
            
colours = ("r", "g","b")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for iclass in range(len(pie)):
       ax.scatter(X[iclass][0], X[iclass][1], X[iclass][2], c=colours[iclass])
plt.show()
#main()
#print(tra)