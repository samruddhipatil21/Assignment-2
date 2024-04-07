# Assignment-2
This script utilizes PyTorch's Lightning library to construct a Convolutional Neural Network (CNN) for image classification tasks, employing the iNaturalist 12K dataset. The CNN architecture encompasses five convolutional layers, each succeeded by max-pooling, batch normalization, dropout, and culminates in a fully connected layer with softmax activation. Key model hyperparameters, including activation function, batch normalization, data augmentation, filter organization, and dropout rate, are customizable through command-line arguments.
### Instructions to train and evaluate the model
1. Install the required libraries:
```python
!pip install pytorch_lightning
!curl -SL https://storage.googleapis.com/wandb_datasets/nature_12K.zip > Asg2_Dataset.zip
!unzip Asg2_Dataset.zip
!pip install wandb
```
2. Give proper path for the dataset.
3. To train the model run train.py using the below command: 
```python
python train.py --wandb_entity myname --wandb_project myprojectname
```
Following are the supported command line arguments:

|           Name           | Default Value | Description                                                               |
| :----------------------: | :-----------: | :------------------------------------------------------------------------ |
| `-wp`, `--wandb_project` | myprojectname | Project name used to track experiments in Weights & Biases dashboard      |
|  `-we`, `--wandb_entity` |     myname    | Wandb Entity used to track experiments in the Weights & Biases dashboard. |
|  `-e`, `--epochs` |     10    | Number of epochs to train neural network.. |
|  `-lr`, `--learning_rate` |     0.0001    | Learning rate used to optimize model parameters. |
|  `-fz`, `--filter_size` |    64     | Filter size used by convolution layer. |
|  `-fo`, `--filter_organisation` |    same     | choices= [same,half,double] Filter organisation used by convolution layer. Using same would give the same filter size to all the layers, using half would keep halving the filter size in the successive layers and using double would mean doubling the filter size in the successive layers.|
|  `-da`, `--DataAugmentation` |     No    | choices = [Yes, No]. Perform data augmentation or not |
|  `-bn`, `--batch_normalisation` |     No    | choices = [Yes, No]. Perform batchNormalisation or not |
|  `-do`, `--drop_out` |     0.3    | Dropout value. |
|  `-a`, `--activation_function` |     GELU    | choices = [RELU, GELU, SELU, MISH]. Activation function to use |
\

### Dataset and Data Loaders
The iNaturalist 12K dataset is loaded from the \
/kaggle/input/inaturalist12k/Data/inaturalist_12K/train  and \
/kaggle/input/inaturalist12k/Data/inaturalist_12K/val \
directories for training and testing (if we are using kaggle), respectively. Depending on the value of the data_augmentation parameter in the project configuration, either transform or transform_augmented is applied to the training set. The testing set always uses transform.

### Data Transformations
Two sets of data transformations are defined: transform and transform_augmented. Both transform the images to have a size of 256x256 pixels and convert them to tensors. However, transform_augmented applies additional data augmentation techniques to the images, including random cropping, flipping, and rotating. Both transformations then normalize the images using the mean and standard deviation values for the ImageNet dataset.

# Set up the configuration for the sweep using the `wandb.sweep` function
sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'val accuracy', 'goal': 'maximize'},
    'parameters': {
        'drop_OUT': {"values": [0.2, 0.3]},
        "activation_FUN": {
              "values": [ "ReLU", "SiLU", "GELU", "Mish"]
          },

          "learning_rate": {
              "values": [1e-3, 1e-4]
          }
          ,
        "filter_ORG":{
            #"values":["[4,8,16,32,64],[64,32,16,8,4],[32,32,32,32,32],[64,64,64,64,64]"]
      
        "values": [
                [4,8,16,32,64],[64,32,16,8,4],[32,32,32,32,32],[64,64,64,64,64]
            ]
          },
        "data_AUG":{
            "values":["Yes","No"]
        },
        "batch_NORM":{
            "values":["Yes","No"]
        },
          "epochs": {
              "values": [5, 10]
          },
    }
}
