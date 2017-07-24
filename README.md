# Tensorflow_AgeGender_CNN
Implement the CNN using tensorflow to recognize age or gender. Training with IMDB_WIKI. 
Some modifications here
1. Training with IMDB_WIKI set with image size 227 * 227. To match the size of CNN, the convolution layer applied padding.
2. Using Relu instead of Sigmod as activation function.
3. Applied dropout in the FC layer.