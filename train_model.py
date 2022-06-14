# Pytorch dependencies
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
from torchvision import datasets, transforms

#parser for reading command-line arguments
import argparse
import time
import os
import json

# import SageMaker debugger client library
import smdebug.pytorch as smd

"""
# Import dependencies for Debugging and Profiling
from smdebug import modes
from smdebug.profiler.utils import str2bool
from smdebug.pytorch import get_hook
"""


def test(model, test_loader, criterion, device, hook):
    '''
        This function takes a model and a testing data loader and get the
        test accuray/loss of the model.
        Remember to include any debugging/profiling hooks that you might need
    '''
    
    # set model to EVAL mode
    model.eval() #because finetuning (pre-trained)
    
    # set SMDebug hook to EVAL mode
    hook.set_mode(smd.modes.EVAL) ### SageMaker debugger
    
    correction = 0
    loss_counter = 0
    
    with torch.no_grad():
        for image, target in test_loader:
            image = image.to(device)
            target = target.to(device)
            
            output = model(image)
            loss = criterion(output, target)
            pred = torch.argmax(input = output, dim = 1)
            
            loss_counter += loss.item()
            correction += torch.sum(pred == target.data).item()
            
    # print statistics
    print(f"Test Loss {loss_counter/len(test_loader.dataset)}, \ Test Accuracy {100*(correction/len(test_loader.dataset))}%")



def train(model, train_loader, criterion, optimizer, device, hook):
    '''
        This function takes a model and data loaders for training 
        and will get train the model.
        Remember to include any debugging/profiling hooks that you might need
    '''
    
    loss_counter = 0
    
    # set model to TRAIN mode
    model.train()
    
    # set SMDebug hook to TRAIN mode
    hook.set_mode(smd.modes.TRAIN) ### SageMaker debugger

    # training loop
    for batch_n, (image, target) in enumerate(train_loader, 1):
        image = image.to(device)
        target = target.to(device)
        
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        #statistics
        loss_counter += loss.item()
        
    # return loss
    return loss_counter/len(train_loader.dataset)



def net():
    '''
        This function initializes the model.
        Remember to use a pretrained model
    '''

    # download pretrained model
    model = models.resnet18(pretrained = True)
    
    # freeze all the pre-trained layers
    for p in model.parameters():
        p.requires_grad = False
    
    # get the numbers of input features in the last FC layer
    nin_features = model.fc.in_features
    
    # replace the last FC layer to meet the number of classes (out_features) in the actual dataset
    model.fc = nn.Sequential(nn.Linear(in_features = nin_features, out_features = 133)) #oss. there're 133 n_classes in the dataset
    
    return model



def create_data_loaders(data_path, batch_size):
    '''
    This is an optional function that you may or may not need
    depending on whether you need to use data loaders or not
    '''
    
    transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    
    return loader



def save_model(model, model_dir):
    ## Save the trained model
    print('Saving the model!!!')
    modeldir_path = os.path.join(model_dir, "modelp3.pt")
    torch.save(model.state_dict(), modeldir_path)
    for t in range(10):
        print('-', end="", flush=True)
        time.sleep(0.2)
    print(f"Model Dir: {modeldir_path}")



def main(args):
    
    # define device to run the model
    use_cuda = args.num_gpus > 0
    print(f"Number of gpus available - {args.num_gpus}")
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Running on device {device}")
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(f"Running on device {device}")
    
    ## Initialize a model by calling the net function
    model=net()
    
    # create the SMDebug hook
    hook = smd.Hook.create_from_json_file() ### SageMaker debugger
    # register the SMDebug hook to the model to save output tensors
    hook.register_hook(model) ### SageMaker debugger
    
    # move model to GPU memory
    model = model.to(device)
    
    ## Create your loss and optimizer
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = args.lr)
        
    ## Call the train function to start training your model
    # Remember that you will need to set up a way to get training data from S3
    ######## train_loader = create_data_loaders("./data/dogImages/train", args.batch_size) #if locally training
    train_loader = create_data_loaders(args.data_train, args.batch_size)
    for epoch in range(1, args.epochs + 1):
        loss = train(model, train_loader, loss_criterion, optimizer, device, hook)
        # print out training progress
        print("Train Epoch: {}, Average Loss: {}".format(epoch, loss))

    ## Test the model to see its accuracy
    test_loader = create_data_loaders(args.data_test, args.test_batch_size) 
    test(model, test_loader, loss_criterion, device, hook)

    save_model(model, args.model_dir)



if __name__=='__main__':

    parser=argparse.ArgumentParser(description = "define hyperparameters for fine tuning Pytorch CIFAR10 model")
    
    ## Specify training args
    parser.add_argument(
        "--batch_size",
        type=int,
        default=40,
        metavar="N",
        help="training batch size (default: 64)"
    )
    
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=209,
        metavar="N",
        help="test batch size (default: 100)"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        metavar="N",
        help="number of epochs to train (default: 10)"
    )
    
    parser.add_argument(
        "--lr",
        type=float,
        default=0.003,
        metavar="LR",
        help="learning rate (default: 3e-2)"
    )
    
    # get container environment variables
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current_host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"]) ## path to model artifacts
    parser.add_argument('--channel_names', default=json.loads(os.environ['SM_CHANNELS'])) ## names of data channels
    parser.add_argument("--data_train", type=str, default=os.environ["SM_CHANNEL_TRAIN"]) ## path to training data
    parser.add_argument('--data_test', type=str, default=os.environ['SM_CHANNEL_TEST']) ## path to test data
    parser.add_argument('--data_valid', type=str, default=os.environ['SM_CHANNEL_VALID']) ## path to validation data
    parser.add_argument("--data_train_lst", type=str, default=os.environ["SM_CHANNEL_TRAIN_LST"]) ## path to training metadata
    parser.add_argument('--data_test_lst', type=str, default=os.environ['SM_CHANNEL_TEST_LST']) ## path to test metadata
    parser.add_argument('--data_valid_lst', type=str, default=os.environ['SM_CHANNEL_VALID_LST']) ## path to validation metadata
    parser.add_argument("--num_gpus", type=int, default=os.environ["SM_NUM_GPUS"]) ## number of gpus in the current container
   
    # end argument definition
    args=parser.parse_args()
    
    main(args)
