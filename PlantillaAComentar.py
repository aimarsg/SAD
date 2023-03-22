# This is a sample Python script.

# Press Mayús+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
#python PlantillaAComentar.py --algoritmo="KNN" -k 5 -d 2 -f iris.csv -o irisResultados.csv

import getopt
import sys
import numpy as np
import pandas as pd
import sklearn as sk
import imblearn
from imblearn.under_sampling import RandomUnderSampler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import pickle
import csv

from fs.time import datetime_to_epoch

#valores por defecto#
k=1
d=1
K=1
D=1
w=0
W=1
s=1
S=1
p='./'
f="iris.csv"
oFile="output"
algoritmo=""


########################################################

# Build up our result dataset

def calcularMetricasGuardar(clf, writer, parametros, nombre,test, testX, testY):

    # The model is now trained, we can apply it to our test set:

    predictions = clf.predict(testX)
    probas = clf.predict_proba(testX)

    predictions = pd.Series(data=predictions, index=testX.index, name='predicted_value')
    cols = [
                u'probability_of_value_%s' % label
                for (_, label) in sorted([(int(target_map[label]), label) for label in target_map])
            ]
    
    probabilities = pd.DataFrame(data=probas, index=testX.index, columns=cols)

    # Build scored dataset
    results_test = testX.join(predictions, how='left')
    results_test = results_test.join(probabilities, how='left')
    results_test = results_test.join(test['__target__'], how='left')
    results_test = results_test.rename(columns= {'__target__': 'TARGET'})

    i=0
    for real,pred in zip(testY,predictions):
        #print(real,pred)
        i+=1
        if i>5:
            break
    
    
    #fscore = f1_score(testY, predictions, average=None)#clasificacion binaria
    fscore = f1_score(testY, predictions, average='weighted') #clasificacion multiclase
    print("Fscore: "+str(fscore))

    #PARA ESCRIBIR EN EL CSV
    
    writer.writerow([parametros])

    report = classification_report(testY,predictions, output_dict=True) #guardar reporte en una variable
    
    for label, resultados in report.items():
        #writer.writerow([label])
        if type(resultados) == dict: #si el value es un diccionario, recorrer valores
            writer.writerow([label, resultados['precision'], resultados['recall'], resultados['f1-score']])
        else: #si no es un diccionario es solo un valor (accuracy)
            writer.writerow([label, resultados])
            
    # print(confusion_matrix(testY, predictions, labels=[1,0]))
    print(confusion_matrix(testY, predictions))
    writer.writerow([])

    #guardar modelo con pickle
    pickle.dump(clf, open(nombre,'wb'))
    #guardar en una libreria el f-score de cada modelo para saber cual es mejor
    return fscore

########################################################

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('ARGV   :',sys.argv[1:])
    try:
        options,remainder = getopt.getopt(sys.argv[1:],'o:k:d:p:K:D:W:f:h:s:S:',['output=','algoritmo=','k=','d=','K=','D=','W=','s=','S=','path=','file','help'])
    except getopt.GetoptError as err:
        print('ERROR:',err)
        sys.exit(1)
    print('OPTIONS   :',options)

    for opt,arg in options:
        if opt in ('-o','--output'):
            oFile = arg
        elif opt == '-k':
            k = int(arg)
            K = int(arg)
        elif opt ==  '-d':
            d = int(arg)
            D = int(arg)
        elif opt == '-K':
            K = int(arg)
        elif opt ==  '-D':
            D = int(arg)
        elif opt ==  '-W':
            w = int(arg)
            W = int(arg)
        elif opt ==  '-s':
            s = int(arg)
            S = int(arg)
        elif opt ==  '-S':
            S = int(arg)
        elif opt ==  '--algoritmo':
            algoritmo = arg
        elif opt in ('-p', '--path'):
            p = arg
        elif opt in ('-f', '--file'):
            f = arg
        elif opt in ('-h','--help'):
            print(''' -o outputFile \n --algoritmo algoritmo: KNN o DecisionTree \n -p inputFilePath \n -f inputFileName \n
                       parametos KNN \n -k numVecinos \n -d valorP \n -K maxKvalue(opcional) \n -D maxP (opcional) \n -W weights(0:uniform,1:distance)  \n
                       parametros DecisionTree \n -d max_Depth \n -D max_depth valor maximo (opcional) \n -s min_samples_split \n -S min_samples_split valor maximo(opcional) \n -k min_samples_leaf \n -K min_samples_leaf valor maximo (opcional)''')
            exit(1)

    if p == './':
        iFile=p+str(f)
    else:
        iFile = p+"/" + str(f)
    # astype('unicode') does not work as expected

    def coerce_to_unicode(x):
        if sys.version_info < (3, 0):
            if isinstance(x, str):
                return unicode(x, 'utf-8')
            else:
                return unicode(x)
        else:
            return str(x)

    #Abrir el fichero .csv y cargarlo en un dataframe de pandas
    ml_dataset = pd.read_csv(iFile)

    #comprobar que los datos se han cargado bien. Cuidado con las cabeceras, la primera línea por defecto la considerara como la que almacena los nombres de los atributos
    # comprobar los parametros por defecto del pd.read_csv en lo referente a las cabeceras si no se quiere lo comentado

    print(ml_dataset.head(5))
    #print(ml_dataset.columns)

    # coger solo las features deseadas
    #ml_dataset = ml_dataset[COLUMNAS] //drop para quitar



    #dividir las columnas entre numericas / categoriales / text

    #para coger todas las columnas
    categorical_features = [] #ml_dataset.select_dtypes(include='object')
    numerical_features = ml_dataset.select_dtypes(include='number') #coge automaticamente las columnas numericas
    text_features = []
    
    #pasar los columnas categoricas a unicode y las numericas a float
    for feature in categorical_features:
        ml_dataset[feature] = ml_dataset[feature].apply(coerce_to_unicode)

        #pasar de categorial a numerico:
        unique = ml_dataset[feature].unique()
        i = 0
        target_map = {}
        for cat in unique:
            target_map[cat] = i
            i += 1
        ml_dataset['__'+feature+'__'] = ml_dataset[feature].map(str).map(target_map)
        del ml_dataset[feature] 

    for feature in text_features:
        ml_dataset[feature] = ml_dataset[feature].apply(coerce_to_unicode)

    for feature in numerical_features:
        if ml_dataset[feature].dtype == np.dtype('M8[ns]') or (
                hasattr(ml_dataset[feature].dtype, 'base') and ml_dataset[feature].dtype.base == np.dtype('M8[ns]')):
            ml_dataset[feature] = datetime_to_epoch(ml_dataset[feature])
        else:
            ml_dataset[feature] = ml_dataset[feature].astype('double')

    
    #establecer cual va a ser la clase target y pasar de categorial a numerico
    target_map = {'Iris-versicolor': 0, 'Iris-virginica': 1, 'Iris-setosa': 2}
    ml_dataset['__target__'] = ml_dataset['Especie'].map(str).map(target_map) #pasar de categorial a numerico
    del ml_dataset['Especie'] 

    keys = np.array(list(target_map.keys()))

    # Remove rows for which the target is unknown.
    ml_dataset = ml_dataset[~ml_dataset['__target__'].isnull()]
    print(f)
    print(ml_dataset.head(5))

    #dividir train y test
    train, test = train_test_split(ml_dataset,test_size=0.2,random_state=42,stratify=ml_dataset[['__target__']])
    #print(train.head(5))
    print(train['__target__'].value_counts())
    print(test['__target__'].value_counts())

    


    ###########   PREPROCESADO   ###########

    # MISSING VALUES: decidir si eliminar o imputar
    drop_rows_when_missing = []

    impute_when_missing = [{'feature': 'Ancho de sepalo', 'impute_with': 'MEAN'}, 
                           {'feature': 'Largo de sepalo', 'impute_with': 'MEAN'}, 
                           {'feature': 'Largo de petalo', 'impute_with': 'MEAN'}, 
                           {'feature': 'Ancho de petalo', 'impute_with': 'MEAN'}]

    # tratar los missing values a eliminar
    for feature in drop_rows_when_missing:
        train = train[train[feature].notnull()]
        test = test[test[feature].notnull()]
        print('Dropped missing records in %s' % feature)

    # tratar los missing values a imputar
    for feature in impute_when_missing:
        if feature['impute_with'] == 'MEAN':
            v = train[feature['feature']].mean()
        elif feature['impute_with'] == 'MEDIAN':
            v = train[feature['feature']].median()
        elif feature['impute_with'] == 'CREATE_CATEGORY':
            v = 'NULL_CATEGORY'
        elif feature['impute_with'] == 'MODE':
            v = train[feature['feature']].value_counts().index[0]
        elif feature['impute_with'] == 'CONSTANT':
            v = feature['value']
        train[feature['feature']] = train[feature['feature']].fillna(v)
        test[feature['feature']] = test[feature['feature']].fillna(v)
        print('Imputed missing values in feature %s with value %s' % (feature['feature'], coerce_to_unicode(v)))

    # REESCALADO

    rescale_features = {'Ancho de sepalo': 'AVGSTD', 
                        'Largo de sepalo': 'AVGSTD', 
                        'Largo de petalo': 'AVGSTD', 
                        'Ancho de petalo': 'AVGSTD'}
    
    # Escalar las features indicadas usando MINMAX o z-score
    for (feature_name, rescale_method) in rescale_features.items():
        if rescale_method == 'MINMAX':
            _min = train[feature_name].min()
            _max = train[feature_name].max()
            scale = _max - _min
            shift = _min
        else:
            shift = train[feature_name].mean()
            scale = train[feature_name].std()
        if scale == 0.:
            del train[feature_name]
            del test[feature_name]
            print('Feature %s was dropped because it has no variance' % feature_name)
        else:
            print('Rescaled %s' % feature_name)
            train[feature_name] = (train[feature_name] - shift).astype(np.float64) / scale
            test[feature_name] = (test[feature_name] - shift).astype(np.float64) / scale
    
    ###########     DIVIDIR EL TEST EN 2 / DEV Y TEST (train 0.6 / dev 0.2 / test 0.2)      ###########
    #test, testSet = train_test_split(test,test_size=0.5,random_state=42,stratify=test[['__target__']]) #dividir
    #testSet.to_csv('testIris.csv') #guardar csv
    ###################################################################################################

    #X -> FEATURES
    trainX = train.drop('__target__', axis=1) 
    #trainY = train['__target__']

    testX = test.drop('__target__', axis=1)
    #testY = test['__target__']

    #Y -> ETIQUETAS 
    trainY = np.array(train['__target__']) 
    testY = np.array(test['__target__'])

    # UNDERSAMPLING

    # BINARIO
    # RamdomUnderSampler reduce el desequilibrio de clases
    # 0.5 -> la clase mayoritaria es el doble de la minoritaria
    #undersample = RandomUnderSampler(sampling_strategy=0.5)#la mayoria va a estar representada el doble de veces
    #trainXUnder,trainYUnder = undersample.fit_resample(trainX,trainY)
    #testXUnder,testYUnder = undersample.fit_resample(testX, testY)

    # MULTICLASS 
    undersample = RandomUnderSampler(sampling_strategy = "not minority")
    trainXUnder,trainYUnder = undersample.fit_resample(trainX,trainY)
    testXUnder,testYUnder = undersample.fit_resample(testX, testY)

    

    ############    ENTRENAMIENTO   ###########

    modelos = {}
    weights = ['uniform', 'distance']

    with open(oFile+'.csv', mode='w', newline='') as archivo_csv:
        writer = csv.writer(archivo_csv)
        writer.writerow(['', 'Precision', 'Recall', 'F_score'])#cabeceras

        if algoritmo == "KNN" :
        ###### KNN ######
            for l in range (d, D+1): #distancia
                    
                for j in range(k, K+1, 2): # n vecinos
                    
                    for weight in range(w, W+1): # weight

                        # iniciar clasificador KNN
                        clf = KNeighborsClassifier(n_neighbors = j,       #numero de vecinos
                                                    weights = weights[weight], #peso de los vecinos
                                                    algorithm = 'auto',   #
                                                    leaf_size = 30,       #
                                                    p = l)                #parametro de potencia
                        
                        # Para indicar que esta balanceado #auto
                        clf.class_weight = "balanced"

                        # entrenar el modelo
                        clf.fit(trainX, trainY)
                        parametros = "p: "+str(l)+" k: "+str(j)+" Weights: "+weights[weight]
                        nombre = f+"_KNN_k_"+str(j)+"_p_"+str(l)+"_w_"+weights[weight]

                        #obtener metricas y guardar
                        modelos[nombre]=calcularMetricasGuardar(clf, writer, parametros, nombre,test, testX, testY)
                        #guardar en una libreria para saber cual es el mejor



        elif algoritmo == "DecisionTree":
            ###### ARBOL DE DECISION ######

            for l in range (d, D+1, 3): #max depth

                for j in range(k, K+1): #min_samples_leaf (msl)

                    for mss in range(s, S+1): # min_samples_split (mss)
                        if mss<=1:
                            mss = float(mss)
                        #iniciar clasificador ARBOL DE DECISION
                        clf = DecisionTreeClassifier(   random_state = 1337,
                                                        criterion = 'gini',
                                                        splitter = 'best',
                                                        max_depth = l,
                                                        min_samples_leaf = j ,  
                                                        min_samples_split = mss )
                                    
                        # Para indicar que esta balanceado #auto
                        clf.class_weight = "balanced"

                        # entrenar el modelo
                        clf.fit(trainX, trainY)

                        parametros = "Max depth: "+str(l)+" min_samples_leaf: "+str(j)+" min_samples_split: "+str(mss)
                        nombre = f+"_DTree_maxDepth_"+str(l)+"_msl_"+str(j)+"_mss_"+str(mss)

                        #obtener metricas y guardar
                        modelos[nombre]=calcularMetricasGuardar(clf, writer, parametros, nombre,test, testX, testY)
                        #guardar en una libreria para saber cual es el mejor
        
        else:
            print ("ningun algoritmo seleccionado")
            exit(0)


    print("RESULTADOS:")
    print(modelos)
    print("\n modelo con mejor f-score: "+max(modelos, key = modelos.get))
    
print("bukatu da")