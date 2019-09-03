# Amazon SageMaker Scripts for DeepGauge Deployement

This repository contains scripts that allow to train and deploy a convolutional neural network model on AWS Sagemaker and use NEO to optimized those model for deployment on the Edge devices. Steps that are followed in this repositpry are as follows:

- TensorFlow BYOM: Train with Custom Training Script, Compile with Neo, and Deploy on SageMaker

- Construct a script for distributed training

- Create a training job using the sagemaker.TensorFlow estimator

- Deploy the trained model to prepare for predictions (the old way)¶

- Deploy the trained model using Neo

- Deploying the compiled model


We will do deepgauge classification task, but this time we will compile the trained model using the Neo API backend, to optimize for our choice of hardware. Finally, we setup a real-time hosted endpoint in SageMaker for our compiled model using the Neo Deep Learning Runtime.
