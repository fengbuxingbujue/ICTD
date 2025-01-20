# ICTD
Pytorch Implementation of "ICTD: Integrating CNN and Transformer with Diffusion Models for Robust Image Deblurring and Denoising"

<img src="./figure/OverrallFarme.png" width = "800" height = "400" div align=center />

## Abstract
Image deblurring remains a challenging task in computer vision, particularly in complex scenarios involving motion blur and noise. In this paper, we propose CTFD, a novel deblurring network that integrates CNN and Transformer architectures within a diffusion model framework. Our approach utilizes a multi-scale CNN as an initial deblurring module, followed by a Transformer with an Inter-Intra attention mechanism as the diffusion model's reverse denoising module. This design enhances the initial deblurring results with finer details. To further improve robustness, especially against noise, we incorporate an auxiliary convolutional module, DeNet. Extensive experiments on various blur datasets demonstrate that CTFD outperforms classical and traditional deblurring methods, effectively addressing dynamic deblurring in noisy environments. The integration of CNN, Transformer, and diffusion models in CTFD showcases the adaptability and complementarity of these architectures, yielding superior image restoration results."

## Datasets
Download "[GoPro](https://drive.google.com/drive/folders/1BdV2l7A5MRXLWszGonMxR88eV27geb_n?usp=sharing)" dataset or "Hide" datasets into './datasets' and adjust the datasets path </br>
For example: './datasets/GoPro/train/1.png' </br>
For example: './datasets/Hide/test/1.png' </br>
Use the "trainer_onRealBlur.py" file when testing the RealBlur datasets

## Requirements
The model requires the following additional data bags

## Training
First, open: </br>
conf.yml </br>
Then modify the training parameters: change the MODE parameter to 1, point PATH_GT and PATH_IMG parameters to the location where the data set is saved. </br>
If there is a pretraining parameter, change the CONTINUE_TRAINING parameter to True, and then change the PRETRAINED_PATH_INITIAL_PREDICTOR and PRETRAINED_PATH_DENOISER parameters to the pretraining model save path. After the modification is complete, run: </br>
main.py

## Testing
First, open: </br>
conf.yml </br>
Then modify the training parameters: change the MODE parameter to 0, point TEST_PATH_GT and TEST_PATH_IMG parameters to the location where the data set is saved. At the same time, make sure that TEST_INITIAL_PREDICTOR_WEIGHT_PATH and TEST_DENOISER_WEIGHT_PATH two parameters point to the desired training model result</br>
After the modification is complete, run: </br>
main.py

## Pre-trained Models and GoPro-Noisy Dateset
We will upload our pre-trained model parameters GoPro-Noisy Dateset later.

## Statement
We are trying to put our paper ICTD: Integrating CNN and Transformer with Diffusion Models for Robust Image Deblurring and Denoising Submission is at The Visual Computer journal.
