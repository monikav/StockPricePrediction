import numpy as np
import pandas as pd
from keras.utils import to_categorical
from keras.models import model_from_json

def convertLabel(result):
    classVar = np.copy(result)
    classVar[result < 0] = 0
    classVar[result >= 0] = 1
    return classVar

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



modelJson = open('./model.json', 'r')
model = modelJson.read()
modelJson.close()
modelLoaded = model_from_json(model)
modelLoaded.load_weights("./model.h5")
print("Loaded model from disk")

NumTest = 2500
batchSize = 512
testFile = "../input/featureMatrix_1_test.csv"

testBatches = GenerateData(testFile, batchSize)
modelLoaded.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

confidence = modelLoaded.evaluate_generator(testBatches, steps=(int(np.ceil(NumTest / (batchSize)))),
                                            use_multiprocessing=False, verbose=1)
#predictions = modelLoaded.predict_generator(testBatches, steps=(int(np.ceil(NumTest / (batchSize)))), use_multiprocessing=False, verbose=1)
print(confidence)
#print(predictions)
print("%s: %.2f%%" % (modelLoaded.metrics_names[1], confidence[1] * 100))
