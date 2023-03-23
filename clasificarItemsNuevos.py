# This is a sample Python script.

# Press Mayús+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import getopt
import sys
import numpy as np
import pandas as pd
import sklearn as sk
import imblearn
import pickle
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier


def coerce_to_unicode(x):
        if sys.version_info < (3, 0):
            if isinstance(x, str):
                return unicode(x, 'utf-8')
            else:
                return unicode(x)
        else:
            return str(x)



def preprocesado(ml_dataset):
    #ml_dataset = ml_dataset[COLUMNAS] //drop para quitar
    print(ml_dataset.head(5))
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
            #ml_dataset[feature] = datetime_to_epoch(ml_dataset[feature])
            print()
        else:
            ml_dataset[feature] = ml_dataset[feature].astype('double')

    
    #establecer cual va a ser la clase target y pasar de categorial a numerico
    target_map = {'Iris-versicolor': 0, 'Iris-virginica': 1, 'Iris-setosa': 2}
    ml_dataset['__target__'] = ml_dataset['Especie'].map(str).map(target_map) #pasar de categorial a numerico
    #del ml_dataset['Especie'] 

    #keys = np.array(list(target_map.keys()))

    # Remove rows for which the target is unknown.
    ml_dataset = ml_dataset[~ml_dataset['__target__'].isnull()]
    print(f)
    print(ml_dataset.head(5))

    


    ###########   PREPROCESADO   ###########

    # MISSING VALUES: decidir si eliminar o imputar
    drop_rows_when_missing = []

    impute_when_missing = [{'feature': 'Ancho de sepalo', 'impute_with': 'MEAN'}, 
                           {'feature': 'Largo de sepalo', 'impute_with': 'MEAN'}, 
                           {'feature': 'Largo de petalo', 'impute_with': 'MEAN'}, 
                           {'feature': 'Ancho de petalo', 'impute_with': 'MEAN'}]

    # tratar los missing values a eliminar
    for feature in drop_rows_when_missing:
        ml_dataset = ml_dataset[ml_dataset[feature].notnull()]
        print('Dropped missing records in %s' % feature)

    # tratar los missing values a imputar
    for feature in impute_when_missing:
        if feature['impute_with'] == 'MEAN':
            v = ml_dataset[feature['feature']].mean()
        elif feature['impute_with'] == 'MEDIAN':
            v = ml_dataset[feature['feature']].median()
        elif feature['impute_with'] == 'CREATE_CATEGORY':
            v = 'NULL_CATEGORY'
        elif feature['impute_with'] == 'MODE':
            v = ml_dataset[feature['feature']].value_counts().index[0]
        elif feature['impute_with'] == 'CONSTANT':
            v = feature['value']
        ml_dataset[feature['feature']] = ml_dataset[feature['feature']].fillna(v)
        print('Imputed missing values in feature %s with value %s' % (feature['feature'], coerce_to_unicode(v)))

    # REESCALADO

    rescale_features = {'Ancho de sepalo': 'AVGSTD', 
                        'Largo de sepalo': 'AVGSTD', 
                        'Largo de petalo': 'AVGSTD', 
                        'Ancho de petalo': 'AVGSTD'}
    
    # Escalar las features indicadas usando MINMAX o z-score
    for (feature_name, rescale_method) in rescale_features.items():
        if rescale_method == 'MINMAX':
            _min = ml_dataset[feature_name].min()
            _max = ml_dataset[feature_name].max()
            scale = _max - _min
            shift = _min
        else:
            shift = ml_dataset[feature_name].mean()
            scale = ml_dataset[feature_name].std()
        if scale == 0.:
            del ml_dataset[feature_name]
            print('Feature %s was dropped because it has no variance' % feature_name)
        else:
            print('Rescaled %s' % feature_name)
            ml_dataset[feature_name] = (ml_dataset[feature_name] - shift).astype(np.float64) / scale

    return ml_dataset    


model=""
p="./"
preprocesar = False

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('ARGV   :',sys.argv[1:])
    try:
        options,remainder = getopt.getopt(sys.argv[1:],'p:m:f:h:l',['path=','model=','testFile=','help'])
    except getopt.GetoptError as err:
        print('ERROR:',err)
        sys.exit(1)
    print('OPTIONS   :',options)

    for opt,arg in options:
        if opt in ('-p','--path'):
            p = arg
        elif opt in ('-f', '--file'):
            f = arg
        elif opt in ('-m', '--model'):
            m = arg
        elif opt in ('-l'):
            preprocesar = True
        elif opt in ('-h','--help'):
            print(' -p modelAndTestFilePath \n -m modelFileName -f testFileName\n -l preprocesar test')
            exit(1)

    
    if p == './':
        model=p+str(m)
        iFile = p+ str(f)
    else:
        model=p+"/"+str(m)
        iFile = p+"/" + str(f)
        

    #Abrir el fichero .csv con las instancias a predecir y que no contienen la clase y cargarlo en un dataframe de pandas para hacer la prediccion
    y_test=pd.DataFrame()
    testX = pd.read_csv(iFile)

    if preprocesar :
        testX = preprocesado(testX)

    target=[]
    #guardar los valores de target en una lista para comprobar luego con las predicciones y comprobar los aciertos
    target = testX['__target__'].values.tolist()
    #eliminar las columnas que sobran para el modelo /en este caso el target y la autonumerica
    testX = testX.drop('__target__', axis=1)
    #testX = testX.drop(testX.columns[0], axis=1)

    print(testX.head(5))
    clf = pickle.load(open(model, 'rb'))
    predictions = clf.predict(testX)
    probas = clf.predict_proba(testX)
    y_test['preds'] = predictions
    predictions = pd.Series(data=predictions, index=testX.index, name='predicted_value')
    results_test = testX.join(predictions, how='left')

    #guardar test con predicciones en un csv
    results_test.to_csv('testConPredicciones.csv') #guardar csv

    predic = results_test['predicted_value'].values.tolist()
    print(results_test)

    
    #comparar valores esperados y predecidos 
    # NO hacer si no hay valores esperados
    if len(target)==0:
        f1score = f1_score(target, predictions, average='micro')
        print("F1-Score: "+f1score)

        contador = 0

        # Recorremos las dos listas simultáneamente con un bucle for y comparamos los valores
        #for i in range(len(target)):
        #    if target[i] == predic[i]:
        #        contador += 1
        
        #print(contador/len(target))

    
