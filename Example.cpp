

#include "pch.h"
#include "DeepNet.cpp"



const float learningRate = 0.01;
const int numToLearn = 20000;
const int numIn = 5;
const std::string fileName = "";

void testNetwork(DeepNet *net);

void predict(std::vector<float> in, DeepNet *net);


int main() {
	if (fileName != "") {
		DeepNet net(fileName);

		testNetwork(&net);
	}
	else {
		DeepNet net(numIn, numIn, learningRate, act::tanh);
		net.addHiddenLayer(10, act::tanh);

		std::vector<std::vector<float>> input;
		std::vector<std::vector<float>> output;

		input.reserve(numToLearn);
		output.reserve(numToLearn);

		for (int i = 0; i < numToLearn; i++) {
			std::vector<float> in;
			in.reserve(numIn);

			int index = 0;
			float maxVal = 0;
			for (int j = 0; j < numIn; j++) {
				float val = (float)(rand() % 100) / 100;
				in.push_back(val);
				if (val > maxVal) {
					maxVal = val;
					index = j;
				}

			}



			std::vector<float> out;
			out.reserve(numIn);

			for (int j = 0; j < numIn; j++) {
				out.push_back(index == j ? 1 : 0);
			}

			input.push_back(in);
			output.push_back(out);


		}

		net.learn(input, output);
		

		testNetwork(&net);


		net.writeToFile("net_max.txt");
	}
	

	return -1;
}


void testNetwork(DeepNet *net) {
	for (int i = 0; i < numIn; i++) {
		std::vector<float> test;
		for (int j = 0; j < numIn; j++) {
			test.emplace_back(i == j ? 1 : 0);
		}
		predict(test, net);
	}
}

void predict(std::vector<float> in, DeepNet *net) {
	std::cout << "Prediction";
	for (int i = 0; i < in.size(); i++) {
		std::cout << " " << in[i];
	}
	std::cout << ":";

	std::vector<float> pred = net->predict(in);

	for (int i = 0; i < pred.size(); i++) {
		std::cout << "\t" <<(double)pred[i];
	}
	std::cout << "\n";
}