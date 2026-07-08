# Alternate Training through the Epochs (ATE) for Multi-Task Neural Networks

by [Stefania Bellavia](https://www.researchgate.net/profile/Stefania-Bellavia), 
[Francesco Della Santa](https://www.researchgate.net/profile/Francesco-Della-Santa), and 
[Alessandra Papini](https://www.researchgate.net/profile/Alessandra-Papini-2).

In this repository, we publish the code used to implement the _Alternate Training through the Epochs_ 
(**ATE**) procedure for training _Multi-Task Neural Networks_ (**MTNN**) presented in 
_ATE-SG: alternate through the epochs stochastic gradient for multi-task neural networks_, 
Optimization Methods and Software, 2026, https://doi.org/10.1080/10556788.2026.2659033 [Open Access].

In the article we introduce novel alternate training procedures for hard-parameter sharing MTNNs (see Figure 1, 
below). Traditional MTNN training faces challenges in managing conflicting loss gradients, often yielding sub-optimal 
performance. The proposed alternate training method updates shared and task-specific weights alternately, exploiting 
the multi-head architecture of the model. 
This approach reduces computational costs, enhances training regularization, and improves generalization. 
Convergence properties similar to those of the classical stochastic gradient method are established. 
Empirical experiments demonstrate delayed overfitting, improved prediction, and reduced computational demands. 
In summary, our alternate training procedures offer a promising advancement for the training of hard-parameter 
sharing MTNNs.

![Hard-parameter sharing MTNN](https://raw.githubusercontent.com/Fra0013To/ATEforMTNN/main/NNmultitaskGeneral_ex.png)

Fig. 1: Example of a hard-parameter sharing MTNN with 2 tasks. On the left half of the figure 
there is the input layer in green; on the right half of the figure there are the two output layers in red.

## Table of Contents
- [License](https://github.com/Fra0013To/ATEforMTNN/blob/main/README.md#license)
- [Requirements](https://github.com/Fra0013To/ATEforMTNN/blob/main/README.md#requirements)
- [Getting Started](https://github.com/Fra0013To/ATEforMTNN/blob/main/README.md#getting-started)
    - [Model Creation](https://github.com/Fra0013To/ATEforMTNN/blob/main/README.md#model-creation)
    - [Model Compilation](https://github.com/Fra0013To/ATEforMTNN/blob/main/README.md#model-compilation)
    - [Training the Model with ATE Procedure](https://github.com/Fra0013To/ATEforMTNN/blob/main/README.md#training-the-model-with-ate-procedure)
    - [Run the Example](https://github.com/Fra0013To/ATEforMTNN/blob/main/README.md#run-the-examples)
- [Citation](https://github.com/Fra0013To/ATEforMTNN/blob/main/README.md#citation)

## License
_ATEforMTNN_ is released under the MIT License (refer to 
the [LICENSE file](https://github.com/Fra0013To/ATEforMTNN/blob/main/LICENSE) for details).

## Requirements
- numpy
- tensorflow 2
- matplotlib _(for running the example)_

## Getting Started
The ATE-Stochastic Gradient (ATE-SG) training procedure can be used defining a model through the custom class 
_AlternateTrainingEpochsModel_ (from now on, simply called _ATEModel_), subclass of tensorflow.keras.models.Model, and 
running the "_.ate_fit_" method.

### Model Creation

An ATEModel object is created exactly as any other hard-parameter sharing MTNN in keras, but the _ATEModel_ clas must 
be used and the trunk's layer names must be characterized by the same prefix. 

All the model's layers having a name that 
doesn't start with this prefix are considered branch layers (i.e., task-specific layers). 

The input arguments for initializing an ATEModel object are the same ones used for the
_tensorflow.keras.models.Model_ superclass, with only one extra input argument:
_prefix_layername_shared_ (default value: "_trunk_").

### Model Compilation

There are no differences in compiling ATEModel objects.

**ATTENTION (SGD OPTIMIZER):** for running an ATE-SG procedure, the ATEModel object _must be compiled using the keras SGD Optimizer_. 
Using other optimizers the ATE procedure still works (using the chosen optimizer), 
but convergence properties are not guaranteed. 
The convergence theoretical results of the paper are guaranteed only for the ATE-SG procedure (for now).


### Training the Model with ATE Procedure
Once the ATEModel object is compiled, the ATE training procedure can be executed just calling the _.ate_fit_ method.
This method has the same input arguments of the _.fit_ method except for the following things:
- **extra input arguments:** 
    * _epochs_shared_: integer. The number of consecutive epochs dedicated to the update of the shared parameters ($E_0$ in the paper);
    * _epochs_taskspecific_: integer. The number of consecutive epochs dedicated to the update of the task-specific parameters ($E_{\mathrm{ts}}$ in the paper);
    * _verbose_alternate_: boolean. Argument for printing on screen information about which kind of parameters (shared or task-specific) are training.
- **callbacks:** the functionality of the keras callbacks _TerminateOnNaN_, _ReduceLROnPlateau_, and _EarlyStopping_ have been correctly extended in order to work with the ATE procedure. On the other hand, functionalities of other callbacks are not guaranteed (for now). 


### Run the Examples
To see a code example of ATE-SG procedure, see the script 
[example_alternate_training_epochs.py](https://github.com/Fra0013To/ATEforMTNN/blob/main/example_alternate_training_epochs.py)
in this repository.

To run the example (bash terminal):
1. Clone the repository:
    ```bash 
    git clone https://github.com/Fra0013To/ATEforMTNN.git
    ```
2. Install the [required python modules](https://github.com/Fra0013To/ATEforMTNN/blob/main/README.md#requirements).
    ```bash
    pip install numpy
    pip install matplotlib
    pip install tensorflow==2.X.Y
    ```
   
    where _tensorflow==2.X.Y_ denotes a generic version of tensorflow 2.
    
3. Run the script [example_alternate_training_epochs.py](https://github.com/Fra0013To/ATEforMTNN/blob/main/example_alternate_training_epochs.py)
for the ATE-SG procedure example:
    ```bash
    python example_alternate_training_epochs.py
    ```

![Output Plot](https://raw.githubusercontent.com/Fra0013To/ATEforMTNN/main/exampleATE_testset.png)

Fig. 2: Plot returned by running the example script. Dots are the test data. The more transparent they are, 
the more the predicted value is near to the threshold value (i.e., 0.5). The background is colored according to the
correct classification of the domain.


## Citation
If you find the Alternate Tranining for MTNNs useful in your research, please cite:
#### BibTeX
> @article{Bellavia28042026,  
> author = {Stefania Bellavia and Francesco Della Santa and Alessandra Papini},  
> title = {ATE-SG: alternate through the epochs stochastic gradient for multi-task neural networks},  
> journal = {Optimization Methods and Software},  
> volume = {0},  
> number = {0},  
> pages = {1--33},  
> year = {2026},  
> publisher = {Taylor \& Francis},  
> doi = {10.1080/10556788.2026.2659033},  
> URL = {https://doi.org/10.1080/10556788.2026.2659033},  
> eprint = {https://doi.org/10.1080/10556788.2026.2659033}  
> }
#### RIS
> TY  - JOUR  
> AU  - Bellavia, Stefania  
> AU  - Della Santa, Francesco  
> AU  - Papini, Alessandra  
> TI  - ATE-SG: alternate through the epochs stochastic gradient for multi-task neural networks  
> JO  - Optimization Methods and Software  
> VL  - 0  
> IS  - 0  
> SP  - 1  
> EP  - 33  
> PY  - 2026  
> DA  - 2026/04/28  
> PB  - Taylor & Francis  
> DO  - 10.1080/10556788.2026.2659033  
> UR  - https://doi.org/10.1080/10556788.2026.2659033  
> ER  -  

## Update
- 2026.05.01: Repository update after publication of the paper.
- 2024.01.02: Repository creation (preprint 2023, avaible at https://doi.org/10.48550/arXiv.2312.16340).
