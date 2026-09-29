## ResNet-18 Implementation

Trained a resnet-18 based architecure model on cifar-10 dataset with custom Residual Blocks. 

---

## Architecture

Below is a visual of the model's architecture:

<img src="Resnet_architecture.png" width="25%" alt="My Image">

The residual blocks in the diagram have the following structure within them:
- 2 Convolution layers
- 2 Batch Normalization layers
- 2 RELU
- 1 Skip connection

Skip connection here means that data skips passing through some layers and gets added to the output.

---

## Deployed App

Try out the model here:

<a href="https://huggingface.com/spaces/ahmad-777/ResNet-Implementation"><img src="https://img.shields.io/badge/HuggingFace-ahmad--777%2FResNet--Implementation-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Hugging Face Space"/></a>

---

## License

- MIT 