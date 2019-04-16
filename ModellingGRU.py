import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Bidirectional
from keras.utils import to_categorical
from keras.layers import BatchNormalization
from keras.layers import GRU
from keras import optimizers
import matplotlib.pyplot as plt

def GenerateData(fileLoc, batchSize):
    batchSize = batchSize
    while True:
        for batch in pd.read_csv(fileLoc, chunksize=batchSize, sep=" "):
            indepFeat = np.array(batch.ix[:,:-1])
            indepFeat = np.reshape(indepFeat,(indepFeat.shape[0],30,100))
            depFeat = np.matrix(batch.ix[:,-1]).T
            depFeat = to_categorical(convertLabel(depFeat), num_classes=2).astype("int")
            depFeat= np.matrix(depFeat)
            yield indepFeat,depFeat

def convertLabel(result):
    classVar = np.copy(result)
    classVar[result < 0] = 0
    classVar[result >= 0] = 1
    return classVar

def buildGRU():
    model = Sequential()
    model.add(Bidirectional(GRU(128, return_sequences=True) ,input_shape=(30, 100),merge_mode ='ave'))
    model.add(Bidirectional(GRU(64, return_sequences=True,activation='relu'),merge_mode ='ave'))
    model.add(BatchNormalization(axis=-1))
    model.add(Flatten())
    model.add(Dense(512,activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(Dropout(0.2))
    model.add(Dense(16,activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(Dense(2, activation='softmax'))
    adam = optimizers.adam(lr=0.0003)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['acc'])
    return model

def main():
    model = buildGRU()
    print(model.summary())
    batchSize = 512
    trainingData = "input/featureMatrix_0_validation.csv"
    validationData = "input/featureMatrix_1_validation.csv"
    testData = "input/featureMatrix_1_test.csv"
    numTrainRecords = 100000
    numValidRecords = 25000
    numTestRecords = 2500

    trainingDataBatch = GenerateData(trainingData, batchSize)
    validationBatchData =GenerateData(validationData, batchSize)
    batchTestData = GenerateData(testData, batchSize)
    batchTestData1 = GenerateData(testData, batchSize)

    modelProgress = model.fit_generator(generator=trainingDataBatch,
                      steps_per_epoch=(int(np.ceil(numTrainRecords/ (batchSize)))),
                      epochs=5,
                      verbose=1,
                      validation_data = validationBatchData,
                      validation_steps = (int(np.ceil(numValidRecords // (batchSize)))),
                      max_queue_size=32)

    confidence = model.predict_generator(batchTestData1, steps=(int(np.ceil(numTestRecords / (batchSize)))),
                                   use_multiprocessing=False, verbose=1)
    print(confidence)
    confidence = model.evaluate_generator(batchTestData, steps=(int(np.ceil(numTestRecords / (batchSize)))),
                                    use_multiprocessing=False, verbose=1)
    print(confidence)
    print("%s: %.2f%%" % (model.metrics_names[1], confidence[1]*100))

    modelDump = model.to_json()
    with open("./input/model.json", "w") as json_file:
        json_file.write(modelDump)
    model.save_weights("./input/model.h5")
    print("Saved model to disk")
    plt.plot(modelProgress.history['loss'])
    plt.plot(modelProgress.history['val_loss'])
    plt.title('model_loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(['train','validation'],loc='upper left')
    plt.show()

main()
