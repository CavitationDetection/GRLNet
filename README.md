# Regional-Local Adversarially Learned One-Class Classifier Anomalous Sound Detection in Global Long-Term Space

## Requirements

- Python 3.8.11
- torch 1.9.1
- torchvision 0.10.1

Note: our model is trained on NVIDIA GPU (A100).

## Code execution

- train.py is the entry point to the code.
- main.py is the main function of our model.
- network.py is the network structure of local generator, regional generator and the discirminator of our method.
- opts.py is all the necessary parameters for our method (e.g. comprehensive output factor, learning rate and data loading path and so on).
- Execute train.py
