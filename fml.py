import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import itertools

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

df = pd.read_csv('./heart.csv')
df

df.isnull().sum()

df[df.duplicated()]

df.drop_duplicates(keep='first', inplace=True)
df[df.duplicated()]

df.reset_index(drop=True)
df

df.describe()

df_corr_mat = df.corr()
px.imshow(df_corr_mat)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape)
print(X_test.shape)

print(y_train.shape)
print(y_test.shape)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def get_models(num_layers: int,
               min_nodes_per_layer: int,
               max_nodes_per_layer: int,
               node_step_size: int,
               input_shape: tuple,
               hidden_layer_activation: str = 'relu',
               num_nodes_at_output: int = 1,
               output_layer_activation: str = 'sigmoid') -> list:
  
  node_options = list(range(min_nodes_per_layer, max_nodes_per_layer + 1, node_step_size))
  layer_possibilities = [node_options] * num_layers
  layer_node_permutations = list(itertools.product(*layer_possibilities))
  
  models = []
  for permutation in layer_node_permutations:
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
    model_name = ''

    for nodes_at_layer in permutation:
      model.add(tf.keras.layers.Dense(nodes_at_layer, activation=hidden_layer_activation))
      model_name += f'dense{nodes_at_layer}_'

    model.add(tf.keras.layers.Dense(num_nodes_at_output, activation=output_layer_activation))
    model._name = model_name[:-1]
    models.append(model)
      
  return models

all_models = get_models(
  num_layers=3, 
  min_nodes_per_layer=16, 
  max_nodes_per_layer=256, 
  node_step_size=16, 
  input_shape=(13,),
  hidden_layer_activation='LeakyReLU'
)

all_models[0].save(os.path.join('.', 'models', 'test.h5'))

def optimize(models: list,
            X_train: np.array,
            y_train: np.array,
            X_test: np.array,
            y_test: np.array,
            epochs: int = 50,
            verbose: int = 0) -> pd.DataFrame:
  
  results = []
  
  def train(model: tf.keras.Sequential) -> dict:
    model.compile(
      loss=tf.keras.losses.binary_crossentropy,
      optimizer=tf.keras.optimizers.Adam(),
      metrics=[
        tf.keras.metrics.BinaryAccuracy(name='accuracy')
      ]
    )
    
    model.fit(
      X_train,
      y_train,
      epochs=epochs,
      verbose=verbose
    )
    
    preds = model.predict(X_test)
    prediction_classes = [1 if prob > 0.5 else 0 for prob in np.ravel(preds)]
    
    return {
      'model_name': model.name,
      'test_accuracy': accuracy_score(y_test, prediction_classes),
      'test_precision': precision_score(y_test, prediction_classes),
      'test_recall': recall_score(y_test, prediction_classes),
      'test_f1': f1_score(y_test, prediction_classes)
    }
  
  index = 0
  for model in models:
    try:
      print(model.name, end=' ... ')
      res = train(model=model)
      results.append(res)
      model.save(os.path.join('.', 'models', f'{index}_{model.name}.h5'))
    except Exception as e:
      print(f'{model.name} --> {str(e)}')
    index += 1
      
  return pd.DataFrame(results)

optimization_results = optimize(
    models=all_models,
    X_train=X_train_scaled,
    y_train=y_train,
    X_test=X_test_scaled,
    y_test=y_test
)

test = tf.keras.models.load_model(os.path.join('.', 'models', '0_dense16_dense16_dense16.h5'))
test_pred = test.predict(X_test)
test_prediction_classes = [1 if prob > 0.5 else 0 for prob in np.ravel(test_pred)]

curr_pred = all_models[0].predict(X_test)
curr_prediction_classes = [1 if prob > 0.5 else 0 for prob in np.ravel(curr_pred)]
print(precision_score(y_test, test_prediction_classes))
print(precision_score(y_test, curr_prediction_classes))

optimization_results_precision = optimization_results.sort_values(by='test_precision', ascending=False)
optimization_results_precision = optimization_results_precision.reset_index()
optimization_results_precision

optimization_results_recall = optimization_results.sort_values(by='test_recall', ascending=False)
optimization_results_recall = optimization_results_recall.reset_index()
optimization_results_recall

optimization_results_accuracy = optimization_results.sort_values(by='test_accuracy', ascending=False)
optimization_results_accuracy = optimization_results_accuracy.reset_index(drop=True)
optimization_results_accuracy

plt.plot(optimization_results['test_precision'][:])
plt.show()

plt.plot(optimization_results_precision['test_precision'])
plt.show()

print(optimization_results_precision['test_precision'].shape)

plt.plot(optimization_results_recall['test_recall'])
plt.show()

def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          title=None):

    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


    if sum_stats:
        accuracy  = np.trace(cf) / float(np.sum(cf))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            precision = cf[1,1] / sum(cf[:,1])
            recall    = cf[1,1] / sum(cf[1,:])
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy,precision,recall,f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""


    if figsize==None:
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        categories=False

    plt.figure(figsize=figsize)
    sns.heatmap(cf,annot=box_labels,fmt="",cbar=cbar,xticklabels=categories,yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)
    
    if title:
        plt.title(title)

preds = all_models[2109].predict(X_test_scaled)
prediction_classes = [1 if prob > 0.5 else 0 for prob in np.ravel(preds)]
cf_matrix = confusion_matrix(y_test, prediction_classes)

labels = ['True Neg','False Pos','False Neg','True Pos']
categories = ['Zero', 'One']
make_confusion_matrix(cf_matrix, group_names=labels, categories=categories)