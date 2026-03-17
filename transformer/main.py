import os, sys
import torch
import torch.nn as nn
from utilsData import datasetForTox
from models import toxCLFModel
from utilsTrain import getDataLoader, train_one_epoch
from utilsEval import evaluate_model

################################################
################################################
def main():
    dataFileTrain = "../data/dfTrain.csv"
    dataFileValidate = "../data/dfValidate.csv"
    dataFileTest = "../data/dfTest.csv"
    CONFIG_DATA = {
        'separator': '\t',
        'commentColumn': 'comment_text',
        'toxColumn': 'toxicity',
        'toxThreshold': 0.5
    }
    num_epochs = 10
    learning_rate = 1.e-1

    # dataset for taining:
    datasetTrain = datasetForTox(dataFile=dataFileTrain,
                                maxWindowSize=None,
                                dataConfig=CONFIG_DATA
                                )
    maxWindowSizeInTrain = datasetTrain.maxWin
    print(f'maxWindowSizeInTrain={maxWindowSizeInTrain}')
    # dataset for validation:
    datasetValidate = datasetForTox(dataFile=dataFileValidate,
                                maxWindowSize=maxWindowSizeInTrain,
                                dataConfig=CONFIG_DATA
                                )
    # dataset for testing:
    datasetTest = datasetForTox(dataFile=dataFileTest,
                                maxWindowSize=maxWindowSizeInTrain,
                                dataConfig=CONFIG_DATA
                                )

    # loader for training, validation and testing:
    dataloaderTrain = getDataLoader(datasetForLoader=datasetTrain, batchSize=16, shuffle=True, dropLast=True, numWorkers=8)
    dataloaderValidate = getDataLoader(datasetForLoader=datasetValidate, batchSize=128, shuffle=False, dropLast=False, numWorkers=8)
    dataloaderTest = getDataLoader(datasetForLoader=datasetTest, batchSize=128, shuffle=False, dropLast=False, numWorkers=8)

    ##############################################################
    ##############################################################
    vocabSize = datasetTrain.tokenizer.n_vocab
    CONFIG_MODEL = {
        'vocabSize': vocabSize,
        'embeddingDim': 32,
        'windowSize': maxWindowSizeInTrain,
        'numHeads': 4,
        'dropout': True,
        'batchFirst': True,
        'FFDim': 64,
        'numLayers': 3,
        'numClasses': 2,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device={device}')

    myModel = toxCLFModel(config=CONFIG_MODEL)
    myModel.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(myModel.parameters(), lr=learning_rate)

    myModel.to(device)

    list_for_selection_metric = []
    artifactsPath = './artifacts'
    os.system(f'mkdir -p {artifactsPath}')
    for epoch in range(1, 1 + num_epochs):
        modelSave = os.path.join(artifactsPath + f'model_epoch{epoch}.pt')
        loss, acc = train_one_epoch(myModel, dataloaderTrain, optimizer, criterion, device, modelSave)

        artifacts_eval = evaluate_model(myModel, dataloaderValidate, device)
        F1_validate = artifacts_eval['metrics']['f1_score']
        AUC_validate = artifacts_eval['metrics']['pr_auc']

        list_for_selection_metric.append([AUC_validate, epoch])

        print(f"\nEpoch: {epoch}, Training Loss: {loss:.4f}, Training Accuracy: {acc:.4f}, F1_validate={F1_validate}, AUC_validate={AUC_validate}")
    
    list_for_selection_metric.sort(reverse=True)
    print('list_for_selection_metric:')
    print(list_for_selection_metric)
    bestEpoch = list_for_selection_metric[0][1]
    print(f'bestEpoch={bestEpoch}')

    # testing with best model:
    bestModelPath = os.path.join(artifactsPath + f'model_epoch{bestEpoch}.pt')

    modelTest = toxCLFModel(config=CONFIG_MODEL)
    modelTest.to(device)

    checkpointTest = torch.load(bestModelPath)
    modelTest.load_state_dict(checkpointTest['model_state_dict'])

    artifacts_test = evaluate_model(modelTest, dataloaderTest, device)
    F1_test = artifacts_test['metrics']['f1_score']
    AUC_test = artifacts_test['metrics']['pr_auc']
    print(f'On testing dataset: f1-score={F1_test}, auc={AUC_test}')

    print('end of main')


##############################################
#### Entry Point:
##############################################
if __name__ == '__main__':
    print('code begins')
    main()
    print('code ends')
