#include "pch.h"

#include <math.h>

#include <iostream>
#include <fstream>
#include <sstream>


#define matrix std::vector<std::vector<float>>

static std::vector<float> matmul(matrix input1, std::vector<float> input2);
static std::vector<float> matadd(std::vector<float> input1, std::vector<float> input2);
static std::vector<float> activate(std::vector<float> input, float(*activation)(float, bool));




class DeepNet {

	int nInput, nOutput;
	std::vector<matrix> connections;
	matrix biases;
	std::vector<float(*)(float, bool)> activations;


	float learningRate = 0.01;
public:
	DeepNet(int nIn, int nOut, float learnRate=0.01,float(*outActivation)(float, bool) = act::relu)
		:nInput(nIn), nOutput(nOut), learningRate(learnRate)
	{
		activations.push_back(outActivation);

		std::vector<float> outputBiases;
		for (int i = 0; i < nOutput; i++) {
			//Random weights from -1 to 1
			outputBiases.emplace_back((float)(rand() % 100 - 50) / 50);
		}
		biases.push_back(outputBiases);

		matrix conMatrix;
		conMatrix.reserve(nOutput);

		for (int i = 0; i < nOutput; i++) {
			std::vector<float> vec;
			vec.reserve(nInput);

			for (int j = 0; j < nInput; j++) {
				//Random connection weights from -1 to 1
				vec.emplace_back((float)(rand() % 100 - 50) / 50);
			}
			conMatrix.push_back(vec);
		}
		connections.push_back(conMatrix);
	}

	DeepNet(std::string fileName) 
	{
		std::ifstream file(fileName);
		std::stringstream sstream;

		while (file >> sstream.rdbuf());

		std::string content = sstream.str();
		std::vector<float> parsed;

		size_t pos = 0;
		std::string token;
		pos = content.find(" ");
		while (pos != std::string::npos) {
			pos = content.find(" ");
			token = content.substr(0, pos);
			float f;
			std::istringstream(token)>>f;
			if (f == 0 && token != "0") {
				activations.push_back(act::stringToFunc(token));
			}
			else
				parsed.push_back(f);

			content.erase(0, pos+1);
		}


		nInput = parsed[0];

		int nLayers = parsed[1];

		biases.reserve(nLayers);

		std::vector<int> nNodes;
		nNodes.reserve(nLayers);

		int index = 2;

		for (int i = 0; i < nLayers; i++) {
			nNodes.emplace_back(parsed[index]);
			index++;
		}



		for (int i = 0; i < nLayers; i++) {
			std::vector<float> bias;
			bias.reserve(nNodes[i]);
			for (int j = 0; j < nNodes[i]; j++) {
				bias.emplace_back(parsed[index]);
				index++;
			}
			biases.push_back(bias);
		}
		


		connections.reserve(nLayers);
		for (int i = 0; i < nLayers; i++) {
			matrix conn;
			conn.reserve(biases[i].size());
			for (int j = 0; j < biases[i].size(); j++) {
				int sizePrev = nInput;
				if (i != 0)
					sizePrev = biases[i - 1].size();
				std::vector<float> connVec;
				connVec.reserve(sizePrev);

				for (int k = 0; k < sizePrev; k++) {
					connVec.push_back(parsed[index]);
					index++;
				}
				conn.push_back(connVec);
			}
			connections.push_back(conn);
		}
	}

	void addHiddenLayer(int size, float(*activation)(float, bool) = act::relu) {
		activations.insert(activations.end() - 1, activation);

		std::vector<float> newVec;
		newVec.reserve(size);

		for (int i = 0; i < size; i++) {
			//Fill with random weights from -1 to 1
			newVec.emplace_back((float)(rand() % 100 - 50) / 50);
		}

		biases.insert(biases.end() - 1, newVec);


		//length of third last element (before the new layer)
		int sizeInput = 0;
		//test if this is the first layer to be added
		if (biases.size() - 3 == -1)
			sizeInput = nInput;
		else
			sizeInput = biases[biases.size() - 3].size();

		matrix newMat;
		newMat.reserve(size);

		for (int i = 0; i < size; i++) {
			std::vector<float> vec;
			vec.reserve(sizeInput);

			for (int j = 0; j < sizeInput; j++) {
				//Random connection weights from -1 to 1
				vec.emplace_back((float)(rand() % 100 - 50) / 50);
			}
			newMat.push_back(vec);
		}

		connections.insert(connections.end() - 1, newMat);


		//connections after the new layer to the end


		matrix conMatrix;

		int sizeOutput = biases[biases.size() - 1].size();
		conMatrix.reserve(sizeOutput);

		for (int i = 0; i < sizeOutput; i++) {
			std::vector<float> vec;
			vec.reserve(size);

			for (int j = 0; j < size; j++) {
				//Random connection weights from -1 to 1
				vec.emplace_back((float)(rand() % 100 - 50) / 50
				);
			}
			conMatrix.push_back(vec);
		}

		//set the last connection new
		connections[connections.size() - 1] = conMatrix;


	}



	void learn(matrix input, matrix solution, int batchsize = 10) {
		std::cout << "Learning: ";
		//For Logging 
		for (int i = 0; i <= log10(input.size()); i++) {
			std::cout << "0";
		}

		for (int iteration = 0; iteration < input.size() / batchsize + 1; iteration++) {
			std::vector<matrix> weightsDelta;
			weightsDelta.reserve(connections.size());

			//Fill weightsDelta with 0
			for (int i = 0; i < connections.size(); i++) {
				matrix mat;
				for (int j = 0; j < connections[i].size(); j++) {
					std::vector<float> vec;
					for (int k = 0; k < connections[i][j].size(); k++) {
						vec.emplace_back(0);
					}
					mat.push_back(vec);
				}
				weightsDelta.push_back(mat);
			}


			matrix biasesDelta;
			biasesDelta.reserve(biases.size());
			for (int i = 0; i < biases.size(); i++) {
				std::vector<float> vec;
				vec.reserve(biases[i].size());
				for (int j = 0; j < biases[i].size(); j++){
					vec.push_back(0);
				}
				biasesDelta.push_back(vec);
			}


			//Fill weightDelta with the wanted values (Backpropagating without changing the neural net)
			for (int batchNum = 0; batchNum < batchsize&&input.size()>iteration * batchsize + batchNum; batchNum++) {

				int index = iteration * batchsize + batchNum;

				//For Logging
				for (int i = 0; i <= log10(index + 1); i++) {
					std::cout << "\b";
				}
				std::cout << index + 1;



				matrix predictedOutValues = predictMatrix(input[index]);
				matrix predictedOutValuesRaw = predictMatrixRaw(input[index]);
				std::vector<float> predictedOutput = predictedOutValues[predictedOutValues.size() - 1];

				matrix deltas;
				deltas.reserve(connections.size());
				for (int i = 0; i < connections.size(); i++) {
					std::vector<float> vec;
					vec.reserve(connections[i].size());
					for (int j = 0; j < connections[i].size(); j++) {
						vec.emplace_back(0);
					}
					deltas.push_back(vec);
				}
				//std::cout << "Computing Deltas" << std::endl;
				for (int layer = weightsDelta.size(); layer > 0; layer--) {
					//std::cout << "Layer: " << layer << std::endl;

					//the last parameter to calculate delta
					std::vector<float> multiplier;

					//Last Layer
					if (layer == weightsDelta.size()) {
						multiplier.reserve(nOutput);

						for (int i = 0; i < nOutput; i++) {
							multiplier.push_back(predictedOutput[i] - solution[index][i]);
						}
					}
					else {//Hidden Layer
						multiplier.reserve(weightsDelta[layer - 1].size());

						for (int j = 0; j < weightsDelta[layer - 1].size(); j++) {
							float value = 0;
							for (int k = 0; k < weightsDelta[layer].size(); k++) {
								value += deltas[layer][k] * predictedOutValues[layer][k] * connections[layer][k][j];
							}
							multiplier.emplace_back(value);
						}
					}


					for (int i = 0; i < multiplier.size(); i++) {
						deltas[layer - 1][i] = (multiplier[i] * activations[layer - 1](predictedOutValuesRaw[layer - 1][i], true));
					}


					for (int neuron = 0; neuron < weightsDelta[layer - 1].size(); neuron++) {
						if (layer != 1) {
							for (int prev = 0; prev < weightsDelta[layer - 2].size(); prev++) {
								weightsDelta[layer - 1][neuron][prev] -= learningRate * deltas[layer - 1][neuron] * predictedOutValues[layer - 2][prev];
							}
						}
						else {
							for (int prev = 0; prev < nInput; prev++) {
								weightsDelta[layer - 1][neuron][prev] -= learningRate * deltas[layer - 1][neuron] * input[index][prev];
							}
						}

					}

					for (int i = 0; i < biasesDelta[layer-1].size(); i++) {
						biasesDelta[layer-1][i] -= learningRate * deltas[layer - 1][i];
					}


				}



			}


			//Applying the learned deltas
			if (!weightsDelta.empty()) {
				for (int layer = 0; layer < connections.size(); layer++) {
					for (int i = 0; i < connections[layer].size(); i++) {
						for (int j = 0; j < connections[layer][i].size(); j++) {
							connections[layer][i][j] += weightsDelta[layer][i][j];
						}

					}
					for (int i = 0; i < biasesDelta[layer].size(); i++) {
						biases[layer][i] += biasesDelta[layer][i];
					}
				}
				
			}

		}
		std::cout << "\n";
	}


	std::vector<float> predict(std::vector<float> input) {
		//Feed forward
		matrix actvalues;
		actvalues.reserve(biases.size());

		for (int i = 0; i < biases.size(); i++) {
			std::vector<float> inputWeights;
			if (i == 0) {
				inputWeights = input;
			}
			else {
				inputWeights = actvalues[i - 1];
			}
			std::vector<float> raws = matadd(matmul(connections[i], inputWeights), biases[i]);
			actvalues.push_back(activate(raws, activations[i]));


		}
		return actvalues[actvalues.size() - 1];
	}

	matrix predictMatrix(std::vector<float> input) {
		//Feed forward


		matrix actvalues;
		actvalues.reserve(biases.size());

		for (int i = 0; i < biases.size(); i++) {
			std::vector<float> inputWeights;
			if (i == 0) {
				inputWeights = input;
			}
			else {
				inputWeights = actvalues[i - 1];
			}
			std::vector<float> raws = matadd(matmul(connections[i], inputWeights), biases[i]);
			//rawvalues.push_back(raws);
			actvalues.push_back(activate(raws, activations[i]));
		}
		return actvalues;
	}

	matrix predictMatrixRaw(std::vector<float> input) {
		//Feed forward


		matrix actvalues;
		actvalues.reserve(biases.size());
		matrix rawvalues;
		rawvalues.reserve(biases.size());

		for (int i = 0; i < biases.size(); i++) {
			std::vector<float> inputWeights;
			if (i == 0) {
				inputWeights = input;
			}
			else {
				inputWeights = actvalues[i - 1];
			}
			std::vector<float> raws = matadd(matmul(connections[i], inputWeights), biases[i]);
			rawvalues.push_back(raws);
			actvalues.push_back(activate(raws, activations[i]));
		}
		return rawvalues;
	}

	void print() {
		std::cout << "\n\n\n\n";
		std::cout << "Input Layers: " << nInput << std::endl;
		std::cout << "Hidden Layers: ";
		for (int i = 0; i < biases.size() - 1; i++) {
			std::cout << biases[i].size() << " ";
		}
		std::cout << std::endl;
		std::cout << "Output Layers: " << nOutput << std::endl;

		std::cout << std::endl;
		std::cout << "Learning Rate: " << learningRate << std::endl;

		std::cout << "\n\n Biases: \n";
		for (int i = 0; i < biases.size(); i++) {
			for (int j = 0; j < biases[i].size(); j++) {
				std::cout << biases[i][j] << "\t";
			}
			std::cout << "\n";
		}

		std::cout << "\n\n Connections: \n";

		for (int i = 0; i < connections.size(); i++) {
			for (int j = 0; j < connections[i].size(); j++) {
				for (int w = 0; w < connections[i][j].size(); w++) {
					std::cout << connections[i][j][w] << "\t";
				}
				std::cout << "\n";
			}
			std::cout << "\n";
		}

	}

	void writeToFile(std::string fileName) {
		std::ofstream file;
		file.open(fileName);

		file << nInput << " " << biases.size() << " ";
		for (int i = 0; i < biases.size(); i++)
			file << biases[i].size()<<" ";

		file << learningRate<<" ";
		for (int i = 0; i < biases.size(); i++) {
			for (int j = 0; j < biases[i].size(); j++) {
				file << biases[i][j] << " ";
			}
		}

		for (int i = 0; i < connections.size(); i++) {
			for (int j = 0; j < connections[i].size(); j++) {
				for (int k = 0; k < connections[i][j].size(); k++) {
					file << connections[i][j][k]<<" ";
				}
			}
		}
		
		for (int i = 0; i < activations.size(); i++) {
			file << act::funcToString(activations[i]) << " ";
		}

		file.close();
	}


};


static std::vector<float> matmul(matrix input1, std::vector<float> input2) {
	std::vector<float> result;
	result.reserve(input1.size());

	for (int i = 0; i < input1.size(); i++) {
		float value = 0;
		for (int j = 0; j < input1[i].size(); j++) {
			value += input1[i][j] * input2[j];
		}
		result.emplace_back(value);
	}

	return result;
}

static std::vector<float> matadd(std::vector<float> input1, std::vector<float> input2) {
	for (int i = 0; i < input1.size(); i++) {
		input1[i] += input2[i];
	}

	return input1;
}

static std::vector<float> activate(std::vector<float> input, float(*activation)(float, bool)) {
	for (int i = 0; i < input.size(); i++) {
		input[i] = activation(input[i], false);
	}

	return input;
}


