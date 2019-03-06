# DeepNetwork
A Deep Network programmed in C++

**This is only for learing purpoes and should not be seen as a professional network**

To initialize the neural network, you have to give the number of input and output layers, the learning rate and the activation function used in the last layer. The code would therefor look like this: `DeepNet net(numInput, numOutput, learningRate, activation)`. To add a hidden layer you have to call `net.addHiddenLayer(numNodes, activation)` with the number of nodes you want to have in this layer and the activation function you want.

To let the Network learn you have to use `net.learn(input, output)`, where input and output are `std::vector<std::vector<float>>` and the length of the inner vector should be the same as the number of input or output nodes and the length of the outside vector should be the same at input and output.

To predict and outcome with a given input you have to use `net.predict(testInput)`, where testInput is a `std::vector<float>` and the function will return another `std::vector<float>` 

To save the trained network use `net.writeToFile(fileName)` and to load it use the constructor `DeepNet net(fileName)`.
