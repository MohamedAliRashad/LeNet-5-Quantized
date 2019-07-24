[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/MohamedAliRashad/LeNet-5-Quantized/blob/master/LeNet5.ipynb)

# LeNet-5-Quantized
This is an implementation for the LeNet-5 Network with a script for quantizing its weights to 8 bit unsigned integer.

<p align="center">
  <img src="https://github.com/MohamedAliRashad/LeNet-5-Quantized/blob/master/LeNet-5.png" width="750">
</p>

## Requirements
- NumPy
- h5py
- protobuf 3.4+
- PyTorch

## Quantization
The idea of quantization was inspired from [keras.js](https://github.com/transcranial/keras-js) with some tweeks.

I was able to quantize the weights of `conv1` and rebuild its output in grayscale format in order to demonstrate the features learned in the first layer.
