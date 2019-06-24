# LeNet-5-Quantized
This is an implementation for the LeNet-5 Network with a script for quantizing its weights to 8 bit unsigned integer

## Requirements
- NumPy
- h5py
- protobuf 3.4+
- keras or PyTorch

## Quantization
The `encoder.py` and `modelpb2.py` was taken from the [keras.js](https://github.com/transcranial/keras-js) but modified to run on python3 because python2 is nearly deperacted now.

Run the quantization process by typing this in the command line
```
./encoder.py -q LeNet_5.h5
```

For more inforamtion, refer to the `encoder.py` [Docs](https://transcranial.github.io/keras-js-docs/conversion/).
