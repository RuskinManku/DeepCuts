# DeepCuts: Single-shot Interpretabily Based Pruning for BERT
This repo contains the code for the project done as a part of the course, CSE 538: Natural Language Processing. In this project, we apply interpretibility techniques like GradCAM and SmoothCAM as a way to measure importance of weights during pruning.

- We use ShrinkBench(https://github.com/JJGO/shrinkbench) as our base code, and implement multiple new methods to explore the strategies proposed in this project. 
- To accomodate our strategies in ShrinkBench, we modify the following files and add the following functions.
```
├── test.py: Driver code for running experiments for multiple compression ratios and strategies. 
├── mask_IOU.py: New file added, to measure the IOU of zeros given two masks.
├── shrinkbench
│   ├── datasets
│   │   ├── datasets.py: Modified this file with functions, SST2DATA, STSBDATA, STSBDL, COLADL, COLADATA, to download data and process them in PyTorch DataLoader
│   ├── experiment
│   │   ├── prune.py: Included functionality for Lottery Ticket Hypothesis(LTH), and gradient calculation over multiple batches.
│   │   └── train.py: We modify this file for tuning hyper-parameters like optimizer, batch size and learning rate.
│   ├── metrics
│   │   ├── accuracy.py: Modified the accuracy measuring function for binary classification, and added measure for MSE loss.
│   ├── models
│   │   ├── BertModelSTSB.py: A new model, that takes a pretrained-bert, and adds a regression head on top for the STSB dataset. 
│   │   ├── BertNet.py: A new model, that takes a pretrained-bert, and adds a classifiar on top for the SST2 dataset and CoLA dataset.
│   ├── pruning
│   │   ├── abstract.py: This is the base class of pruning, we modify it to incorporate LTH, and also modify the constructor to pass parameters used for implementing our strategies.
│   │   ├── mixin.py: The GradientMixin and ActivationMixin functions in this file is used to accumulate gradients and activations, and hence we modify this to store the same across multiple batches.
│   │   ├── utils.py: This is the main function that runs a forward pass and a backward pass, to compute gradients. It is in this function that we introduce a random noise based on the activation of previous layer, for SmoothCAM. 
│   │   └── vision.py: Since this function inherits from the abstract class for pruning, we modify the constructor for the same.
│   ├── strategies
│   │   ├── magnitude.py: Here we implement functions LayerGradCAM, LayerSmoothGrad, LayerSmoothGradCAM, LayerGradCAMShift, LayerSmoothGradCAMShift as a part of strategies proposed in this project.
```
- To run the code, mention the compression ratios and strategies to use in test.py, use and train.py for modifying batch size and learning rate. Below is a sample test.py,
```
from shrinkbench.experiment import PruningExperiment

for  c in [2,3.5,4]:
    for strategy in ['LayerGradCAMShift','LayerSmoothGrad','LayerMagWeight','LayerSmoothGradCAMShift']:
        exp = PruningExperiment(dataset='SST2DATA', 
                                model='BertNet',
                                strategy=strategy,
                                compression=c,
                                train_kwargs={'epochs':10},
                                is_LTH=True)
        exp.run()
```
- Use BertNet with SST2DATA and COLADATA, and BertModelSTSB with STSBDATA. The name for various strategies is,
  - "GlobalMagWeight": Consider only the magnitude of weight for pruning, and threshold globally.
  - "LayerMagWeight": Consider only the magnitude of weight for pruning, and threshold layer-wise.
  - "LayerGradCAM": Consider product of gradient and class activation for pruning, and threshold layer-wise.
  - "LayerSmoothGrad": Consider magnitude and smoothed-out gradients for pruning, and threshold layer-wise.
  - "LayerSmoothGradCAM": Consider smoothed-out gradients along with class activation map for pruning, and threshold layer-wise.
  - "LayerGradCAMShift": Same as LayerGradCAM, but with shifted output maps.
  - "LayerSmoothGradCAMShift": Same as LayerGradCAMShift, but considering smoothed-out gradients.
- Install the following libraries to run the code:
```
transformers==4.21.3
datasets
torchdata==0.5.0
torchtext==0.14.0
pytorch==1.13.0
pandas==1.3.4
matplotlib==3.6.2
numpy==1.23.2
tqdm==4.64.1
```
