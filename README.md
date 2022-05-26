# Coral-TPU
A collection of scripts to compile existing tf (2.0) + keras models to an edge TPU compatible format. Currently tested with the following models.
1. LPRnet
2. MobilenetV2-SSDLite 
## LPRnet 
43 ops mapped to TPU, 32 ops mapped to CPU.

## MobilenetV2-SSDLite (Retrained with TF Object Detection API)
112 ops mapped to TPU, 54 ops mapped to CPU

## Config 
Indicate necessary constants in ``` config.py```

