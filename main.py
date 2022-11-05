# Author Zahra Kharazian, zahkha20@student.hh.se

# Please extract tweets' text using tweets' ID and add it to the dataframe before running this code
# The code is based on RoBERTa model from Huggingface’s transformers library.
# For more information on how to install the required libraries, please visit: https://towardsdatascience.com/tensorflow-and-transformers-df6fceaf57cc

import tensorflow as tf
from transformers import TFAutoModel, AutoTokenizer
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report

# set a random number for seed
tf.random.set_seed(10)
np.random.seed(10)

# before reading the data files, you need to extract the required tweets information using tweets id.
df_ac1 = pd.read_csv('data/Data_AC_1st.csv')
df_ac2 = pd.read_csv('data/Data_AC_2nd.csv')
df_ac3 = pd.read_csv('data/Data_AC_3rd.csv')
df_ac4 = pd.read_csv('data/Data_AC_4th.csv')

df_bc1 = pd.read_csv('data/Data_BC_1st.csv')
df_bc2 = pd.read_csv('data/Data_BC_2nd.csv')
df_bc3 = pd.read_csv('data/Data_BC_3rd.csv')
df_bc4 = pd.read_csv('data/Data_BC_4th.csv')

df = pd.concat([df_ac1, df_ac2, df_ac3, df_ac4, df_bc1, df_bc2, df_bc3, df_bc4])

def add_sum_label_weighted(df1):
    # replace nan with zero
    df1['label_Zahra'] = df1['label_Zahra'].fillna(0)
    df1['label_Hanna'] = df1['label_Hanna'].fillna(0)
    df1['label_Maj'] = df1['label_Maj'].fillna(0)

    df1['label_sum'] = df1['label_Zahra'] + df1['label_Hanna'] + df1['label_Maj']

    # remove gray labels
    df = df1[df1['label_sum'] != 1]

    df['sample_weight'] = df['label_sum']
    df.loc[df['sample_weight'] < 2, 'sample_weight'] = 1
    df['sample_weight'] = df['sample_weight'].astype(int)

    df['target'] = df['label_sum']
    df.loc[df['target'] < 2, 'target'] = 0
    df.loc[df['target'] >= 2, 'target'] = 1
    df['target'] = df['target'].astype(int)
    return df

df = add_sum_label_weighted(df)

df1 = df.copy()

SEQ_LEN = 50

# initialize model and tokenizer
bert = TFAutoModel.from_pretrained("vinai/bertweet-base")
tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")


# define function to handle tokenization
def tokenize(sentence):
    tokens = tokenizer.encode_plus(sentence, max_length=SEQ_LEN,
                                   truncation=True, padding='max_length',
                                   add_special_tokens=True, return_attention_mask=True,
                                   return_token_type_ids=False, return_tensors='tf')
    return tokens['input_ids'], tokens['attention_mask']

# restructure dataset format for BERT
def map_func(input_ids, masks, labels):
    return {'input_ids': input_ids, 'attention_mask': masks}, labels


def map_func2(input_ids_and_masks, labels):
    if labels[0] <= 1 and labels[1] <= 1:
        return input_ids_and_masks, labels
    if labels[0] == 0:
        return input_ids_and_masks, tf.constant([0, 1], dtype='float64')
    return input_ids_and_masks, tf.constant([1, 0], dtype='float64')

def fix_labels(data):
    data2 = data.unbatch().map(map_func2)
    return data2.batch(32)

def to_tensor(df):
    batch_size = 32

    # shuffle data
    df = df.sample(frac=1).reset_index(drop=True)

    DS_LEN = ( df.shape[0]//batch_size ) + 1 # get batched dataset length
    print('@@@@@@@@@@@@@@@@@@@@@@@ DS_LEN:', DS_LEN)
    # create training-validation-test sets
    # we will create a 60_20_20_split
    train_size = int(0.6 * DS_LEN)
    val_size = int(0.20 * DS_LEN)
    test_size = int(0.20 * DS_LEN)

    arr = df['target'].values  # take sentiment column in df as array
    labels = np.zeros((arr.size, 2))  # initialize empty (all zero) label array
    labels[np.arange(arr.size), arr] = 1  # add ones in indices where we have a value

    # make labels weighted so that we can use weights later
    w = df['sample_weight'].values  # take sentiment column in df as array
    for i, a in enumerate(arr):
        labels[i, a] *= w[i]

    # initialize two arrays for input tensors
    Xids = np.zeros((len(df), SEQ_LEN))
    Xmask = np.zeros((len(df), SEQ_LEN))

    # loop through data and tokenize everything
    for i, sentence in enumerate(df['text']):
        Xids[i, :], Xmask[i, :] = tokenize(sentence)

    # create tensorflow dataset object
    dataset = tf.data.Dataset.from_tensor_slices((Xids, Xmask, labels))

    dataset = dataset.map(map_func)  # apply the mapping function

    # shuffle and batch the dataset
    dataset = dataset.shuffle(1000).batch(batch_size)

    train = dataset.take(train_size)
    test = dataset.skip(train_size)
    val = test.skip(test_size)
    test = test.take(test_size)

    # free up space
    del dataset

    ll = []
    ii = []
    mm = []
    ff = train.unbatch().as_numpy_iterator()
    for i in ff:
        if i[1].max() >= 2:
            ll.append(i[1])
            ii.append(i[0]['input_ids'])
            mm.append(i[0]['attention_mask'])
        if i[1].max() == 3:
            ll.append(i[1])
            ii.append(i[0]['input_ids'])
            mm.append(i[0]['attention_mask'])

    labels_weighted = np.vstack([labels[:0,:], ll])
    Xids_weighted = np.vstack([Xids[:0,:], ii])
    Xmask_weighted = np.vstack([Xmask[:0,:], mm])

    dataset_weighted = tf.data.Dataset.from_tensor_slices((Xids_weighted, Xmask_weighted, labels_weighted))

    dataset_weighted = dataset_weighted.map(map_func)  # apply the mapping function

    dataset_weighted = dataset_weighted.shuffle(1000).batch(batch_size)

    train_weighted = train.concatenate(dataset_weighted).shuffle(1000)

    return fix_labels(train), fix_labels(val), fix_labels(test), fix_labels(train_weighted)

train, val, test, train_weighted = to_tensor(df)

#------------------------------------------


def plot_performance():
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    # plt.plot(results.results['test_accuracy'])

    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')

def plot_loss():
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    # plt.plot(results.results['test_loss'])

    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')

def plot_metrics():
    plot_performance()
    plt.figure()
    plot_loss()
    plt.show()

def build_model(training_type):

    # build the model
    input_ids = tf.keras.layers.Input(shape=(50,), name='input_ids', dtype='int32')
    mask = tf.keras.layers.Input(shape=(50,), name='attention_mask', dtype='int32')

    embeddings = bert(input_ids, attention_mask=mask)[0]  # we only keep tensor 0 (last_hidden_state)

    X = tf.keras.layers.GlobalMaxPool1D()(embeddings)  # reduce tensor dimensionality
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Dense(128, activation='relu')(X)
    X = tf.keras.layers.Dropout(0.1)(X)
    y = tf.keras.layers.Dense(2, activation='softmax', name='outputs')(X)  # adjust based on number of sentiment classes

    model = tf.keras.Model(inputs=[input_ids, mask], outputs=y)
    # plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    # freeze the BERT layer
    model.layers[2].trainable = False

    # compile the model
    optimizer = tf.keras.optimizers.Adam(0.01)
    loss = tf.keras.losses.CategoricalCrossentropy()
    acc = tf.keras.metrics.CategoricalAccuracy('accuracy')

    model.compile(optimizer=optimizer, loss=loss, metrics=[acc])

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience= 4, restore_best_weights=True)

    if training_type == 0:
        # and train it
        history = model.fit(train, validation_data=val, epochs=2, callbacks=[callback])
    else:
        # and train it
        history = model.fit(train_weighted, validation_data=val, epochs=2, callbacks=[callback])

    return history, model

history, model = build_model(0)
# plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
plot_metrics()
model.save_weights('final_models2/Fourth-non-weighted')


#  >>>>>>>>>>>>>>>>>>>>>>>>second run

print('>>>>>>>>>>>>>>>>>>>>>>>>>')
print('>>>>>>>>>>>>>>>>>>>>>>>>second run (weighted)')
print('>>>>>>>>>>>>>>>>>>>>>>>>>')
print('>>>>>>>>>>>>>>>>>>>>>>>>>')
print('>>>>>>>>>>>>>>>>>>>>>>>>>')
print('>>>>>>>>>>>>>>>>>>>>>>>>>')

history, model2 = build_model(1)
# plot_model(model2, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
plot_metrics()
model2.save_weights('final_models2/Fourth-weighted')

# get the labels and predictions
predictions2 = np.array([])
predictions = np.array([])
labels = np.array([])
for x, y in test:
    # predictions = np.concatenate([predictions, model.predict(x)])
    predictions = np.concatenate([predictions, np.argmax(model.predict(x), axis=-1)])
    predictions2 = np.concatenate([predictions2, np.argmax(model2.predict(x), axis=-1)])
    labels = np.concatenate([labels, np.argmax(y.numpy(), axis=-1)])

print('_________labels.sum():', labels.sum())

tf.math.confusion_matrix(labels=labels, predictions=predictions).numpy()
tf.math.confusion_matrix(labels=labels, predictions=predictions2).numpy()

target_names = ['non-informative', 'Informative']
report = classification_report(labels, predictions, target_names=target_names, digits=4)
print(report)
cm = tf.math.confusion_matrix(labels=labels, predictions=predictions).numpy()
print(cm)

target_names = ['non-informative', 'Informative']
report = classification_report(labels, predictions2, target_names=target_names, digits=4)
print(report)
cm = tf.math.confusion_matrix(labels=labels, predictions=predictions2).numpy()
print(cm)