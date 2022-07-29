# Project 3: Image Classification using AWS SageMaker (Udacity's AWS Machine Learning Engineer Nanodegree)

## Title
**"Finetuning an Image Classification Model with a custom dataset in AWS SageMaker: script mode (custom training), hyperparameter tuning, model profiling and debugger, and others good ML engineering practices."**


## Project description
Use AWS Sagemaker to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices.


## Project Set Up and Installation
Enter AWS through the gateway in the course and open SageMaker Studio. 
Download the starter files.
Download/Make the dataset available. 


## Dataset
The provided dataset is the dogbreed classification dataset. A dataset containing 133 classes of dog breeds. It is composed of 7649 images of dogs. 6680 images in the training dataset, 836 images in the test dataset and 835 in validation dataset. The dataset is provided by @udacity/active-public-content and can be downloaded from the link https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip


### SageMaker access to data
Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data. 


## Notebook and scripts instruction:
The `train_and_deploy.ipynb` notebook represents the submission script and contains the code to download and preprocess the data and setups to Sagemaker APIs to submit and evaluate the training job.

The training scrips (`hpo.py` and `train_model.py`) contain the model definition and the training and validation code.

More specifically, the hpo.py script is used when performing hyperparameter tuning and contains the code to train the model and save it to S3, while the train_model.py contains code to perform model profiling and debugging.

The inference script (`inference.py`) contains the code to the Sagemaker Model Server functions for deserializing the trained model and load it for inference and for translating an endpoint request to an inference call to the model. This script is passed to entry_point parameter of sagemaker.pytorch.model.PyTorchModel constructor.



## Model and ML framework
The model used is the Resnet18 image classification model.
The Sagemaker ML framework used is PyTorch.
The model was submited and deployed using SageMaker python SDK.



## Hyperparameter Tuning
**TODO**: Describe hyperparameters used

The model hyperparameters are: 'epochs', 'batch_size', 'test_batch_size', and 'lr'.

'epochs': number of passes on the training dataset
'batch_size': the size of each sample of training data to be loaded per iteration
'test_batch_size': the size of each sample of test data to be loaded per iteration
'lr': the learning rate parameter is used to update the model weight values during the gradient descent process. It's multiplied by the derivative of the loss function (dL/dw) to determine the size of the step the model has to make during the search for the best weight values: (w(i) = w(i-1) - lr*[dL/dw])

Hyperparameter tuning was performed on the values of two hyperparameters: 'lr' and 'batch_size'. Using the following configuration:
hyperparameter_ranges = {
    'lr': ContinuousParameter(0.001, 0.01, scaling_type='Logarithmic'),
    'batch_size': CategoricalParameter([40, 60]),
}



## Debugging and Profiling
**TODO**: Give an overview of how you performed model debugging and profiling in Sagemaker

To perform model debugging you need to modify the the submission script and the training script.

- Modifications to the "submission script" include installation of {smdebug} and importing {sagemaker.debugger} libraries and adaptation of 'Pytorch' estimator to include the debugger rules and the debugging configuration.

1- Install smdebug library

`!pip install smdebug`


2- Import the following 'sagemaker.debugger' libraries:

`from sagemaker.debugger import(
    Rule,
    ProfilerRule,
    rule_configs,
    DebuggerHookConfig,
    ProfilerConfig,
    FrameworkProfile,
    CollectionConfig
)`


3- Define the debugger rules to specify the type of debugger evaluation you want monitor:

`rules = [
    ## debugger
    Rule.sagemaker(rule_configs.vanishing_gradient()),
    Rule.sagemaker(rule_configs.overfit()),
    Rule.sagemaker(rule_configs.overtraining()),
    Rule.sagemaker(rule_configs.poor_weight_initialization()),
    Rule.sagemaker(rule_configs.loss_not_decreasing()),
    ## profiler
    ProfilerRule.sagemaker(rule_configs.LowGPUUtilization()),
    ProfilerRule.sagemaker(rule_configs.ProfilerReport())
]`


4- Define the debugger hook configuration:

`hook_config = DebuggerHookConfig(
    s3_output_path=f"s3://{bucket}/{prefix}/debug-output",
    hook_parameters={
        "train.save_interval":"100",
        "eval.save_interval":"10"
    }
)`


- Modifications to the training script include the import of {smdebug.pytorch} library the definition of SMDebug hook and modifications to train() and test() functions to include the SMDebug hook.

1- Import the SMDebug library within the training script as following:

`import smdebug.pytorch as smd`


2- In the main() function of your training script, create the SMDebug hook and register it to the model, as following:

`hook = smd.Hook.create_from_json_file()
hook.register_hook(model)`

Then, pass the hook to the calling of train() and test() functions within you main() fucntion.


3- Finally modify the train() and test() functions to include the SMDebug hook with respective train and eval modes:

`hook.set_mode(smd.modes.TRAIN)`

`hook.set_mode(smd.modes.EVAL)`


### Results
**TODO**: What are the results/insights did you get by profiling/debugging your model?

The debugging output of my model has detected an overtraining issue.



## Model Deployment
The model is deployed by creating an endpoint and invoking the endpoint with some data. The SageMaker SDK inference APIs methods to acomplish this are respectivelly:
model.deploy(), to create the endpoint and endpoint.predict() to query the endpoint.
The method endpoint.predict() takes an image (of bytes object type) as input data and a ContentType "image/jpeg".
The following code will acomplish that (where paylod is the input to query the endpoint):

`
with open("image_path", 'rb') as f:
    payload = f.read()

result = endpoint.predict(
    data = payload,
    initial_args = {'ContentType':"image/jpeg"}
)
`

