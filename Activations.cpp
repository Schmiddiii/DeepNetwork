
#include "pch.h"

#ifndef ACT
#define ACT

#include <math.h>
#include <algorithm>

#define euler 2.71828


namespace act {

	typedef float(*func)(float, bool);



	inline float sigmoid(float input, bool derivative) {
		return derivative ? act::sigmoid(input, false)*(1 - act::sigmoid(input, false)) : (1 / (1 + pow(euler, -input)));
	}

	inline  float relu(float input, bool derivative) {
		return derivative ? input < 0 ? 0 : 1 : input < 0 ? 0 : input;
	}

	inline float lrelu(float input, bool derivative) {
		return derivative ? input < 0 ? 0.01 : 1 : input < 0 ? (0.01*input) : input;
	}

	inline float tanh(float input, bool derivative) {
		return derivative ? 1 - pow(::tanh(input), 2) : ::tanh(input);
	}

	inline float identity(float input, bool derivative) {
		return derivative ? 1 : input;
	}
	
	inline func stringToFunc(::std::string str) {

		if (str == "sigmoid")
			return sigmoid;
		else if (str == "relu")
			return relu;
		else if (str == "lrelu")
			return lrelu;
		else if (str == "tanh")
			return tanh; 
		else if (str == "identity")
			return tanh;

		
	}

	inline ::std::string funcToString(float(*f)(float, bool) ) {
		if (::std::equal((double * )&sigmoid, (double *)&sigmoid+sizeof(double), (double * )f))
			return "sigmoid";
		else if (::std::equal((double *)&relu, (double *)&relu + sizeof(double), (double *)f))
			return "relu";
		else if (::std::equal((double *)&lrelu, (double *)&lrelu + sizeof(double), (double *)f))
			return "lrelu";
		else if (::std::equal((double *)&tanh, (double *)&tanh + sizeof(double), (double *)f))
			return "tanh";
		else if (::std::equal((double *)&identity, (double *)&identity + sizeof(double), (double *)f))
			return "identity";

		return "No function found";
	}


}


#endif