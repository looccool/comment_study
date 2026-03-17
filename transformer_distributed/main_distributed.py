import os, sys
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from utilsData import datasetForTox
from models import toxCLFModel
from utilsTrain import getDataLoader, train_one_epoch
from utilsEval import evaluate_model

################################################
def setup(rank, world_size):
    # Use "gloo" for CPU-based distributed training, and "nccl" for GPU-based
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

################################################
def main_dist(rank, world_size):
    setup(rank, world_size)
    
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
    if rank == 0:
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
    sampler = DistributedSampler(datasetTrain, num_replicas=world_size, rank=rank)
    dataloaderTrain = getDataLoader(datasetForLoader=datasetTrain, batchSize=4, shuffle=False, dropLast=True, numWorkers=8, sampler=sampler)
    if rank == 0:
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
    myModel = DDP(myModel)
    myModel.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(myModel.parameters(), lr=learning_rate)

    # myModel.to(device)

    list_for_selection_metric = []
    artifactsPath = './artifacts'
    if rank == 0:
        os.system(f'mkdir -p {artifactsPath}')
    for epoch in range(1, 1 + num_epochs):
        sampler.set_epoch(epoch)
        
        modelSave = os.path.join(artifactsPath + f'model_epoch{epoch}.pt')
        loss, acc = train_one_epoch(myModel, dataloaderTrain, optimizer, criterion, device, modelSave, rank)

        if rank == 0:
            artifacts_eval = evaluate_model(myModel, dataloaderValidate, device)
            F1_validate = artifacts_eval['metrics']['f1_score']
            AUC_validate = artifacts_eval['metrics']['pr_auc']

            list_for_selection_metric.append([AUC_validate, epoch])
    
            print(f"\nEpoch: {epoch}, F1_validate={F1_validate}, AUC_validate={AUC_validate}")
            
    if rank == 0:
        list_for_selection_metric.sort(reverse=True)
        print('list_for_selection_metric:')
        print(list_for_selection_metric)
        bestEpoch = list_for_selection_metric[0][1]
        print(f'bestEpoch={bestEpoch}')

    if rank == 0:
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

    cleanup()
    print('end of main')


##############################################
#### Entry Point:
##############################################
if __name__ == '__main__':
    print('code begins')
    
    # world_size = 4
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    main_dist(rank, world_size)
    
    print('code ends')
