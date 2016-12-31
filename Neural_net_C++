//-----ANN v3.8-----//
//Wytse Kraal//

#include <iostream>
#include <string>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <fstream>
using namespace std;

float learning_rate = 1.414213562; // a learning rate for the network its now the sqrt of 2 but you can play with it if you want
float momentum = 0.25; // the momentum, you can also tweak this variable if you want

//The weigths for the neural network
float weights[9] = {}; 

//The training date that will be passed thru the neural network
double training_data[4][2] = { { 1, 0 },
			       { 1, 1 },
			       { 0, 1 },
			       { 0, 0 } };
			       
//The anwsers that the neural network should be targeting for
int anwser_data[4] = { 1, 0, 1, 0 };

int bias = 1;
float h1;
float h2;
float error[4];
float output_neuron;
float gradients[9];
float derivative_O1;
float derivative_h1;
float derivative_h2;
float sum_output;
float sum_h1;
float sum_h2;
float update_weights[9];
float prev_weight_update[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
float RMSE_ERROR = 1;
int epoch = 0;
float RMSE_array_error[20000];
float user_input[2];
char choise = 'Y';
bool repeat = false;

////////Prototyping////////
float sigmoid_function(float x);
void calc_hidden_layers(int x);
void calc_output_neuron();
void calc_error(int x);
void calc_derivatives(int x);
void calc_gradient(int x);
void calc_updates();
void update_new_weights();
void calc_RMSE_ERROR();
void generate_weights();
void train_neural_net();
void start_input();
void safe_data();
////////Prototyping////////

int main()
{
	generate_weights();
	train_neural_net();
	safe_data();
	start_input();
	system("pause");
}

void safe_data()
{
	ofstream dataER;
	dataER.open("errorData1.txt");
	for (int i = 0; i < epoch; i++)
	{
		dataER << i << "   " << RMSE_array_error[i] << endl;
	}
	dataER.close();

	ofstream dataER1;
	dataER1.open("weight_data1.txt");
	for (int i = 0; i < 9; i++)
	{
		dataER1 << i << "   " << weights[i] << endl;
	}
	dataER1.close();
}

void start_input()
{
	do
	{
		if (choise == 'Y')
		{
			cout << "enter data 1: "; cin >> user_input[0]; cout << endl;
			cout << "enter data 2: "; cin >> user_input[1]; cout << endl;
			sum_h1 = (user_input[0] * weights[0]) + (user_input[1] * weights[2]) + (bias * weights[4]);
			sum_h2 = (user_input[0] * weights[1]) + (user_input[1] * weights[3]) + (bias * weights[5]);
			h1 = sigmoid_function(sum_h1);
			h2 = sigmoid_function(sum_h2);

			sum_output = (h1 * weights[6]) + (h2 * weights[7]) + (bias * weights[8]);
			output_neuron = sigmoid_function(sum_output);
			cout << "output = " << output_neuron << endl;

			cout << "Again? Y/N"; cin >> choise;
		}
		else
		{
			break;
		}
	} while ((choise == 'Y' || 'y') && (choise != 'n' || 'N'));
}

void train_neural_net()
{
	while (epoch < 20000)
	{
		for (int i = 0; i < 4; i++)
		{
			calc_hidden_layers(i); // input!!
			calc_output_neuron();
			calc_error(i); // input!!
			calc_derivatives(i); // input!!
			calc_gradient(i); // input!!
			calc_updates();
			update_new_weights();
		}
		calc_RMSE_ERROR();
		RMSE_array_error[epoch] = RMSE_ERROR;
		cout << "epoch: " << epoch << endl;
		epoch = epoch + 1;

		//Adding some motivation so if the neural network is not converging after 4000 epochs it will start over again until it converges
		if (epoch > 4000 && RMSE_ERROR > 0.5)
		{
			repeat = true;
			for (int i = 0; i < 9; i++)
			{
				prev_weight_update[i] = 0;
				update_weights[i] = 0;
				gradients[i] = 0;
			}
			for (int i = 0; i < 4; i++)
				error[i] = 0;
			for (int i = 0; i < epoch; i++)
				RMSE_array_error[i] = 0;
			epoch = 0;
			generate_weights();
		}
	}
}

float sigmoid_function(float x)
{
	float sigmoid = 1 / (1 + exp(-x));
	return sigmoid;
}

void generate_weights()
{
	srand(time(NULL));
	for (int i = 0; i < 9; i++)
	{
		int randNum = rand() % 2;
		if (randNum == 1)
			weights[i] = -1 * (double(rand()) / (double(RAND_MAX) + 1.0)); // generate number between -1.0 and 0.0
		else
			weights[i] = double(rand()) / (double(RAND_MAX) + 1.0); // generate number between 1.0 and 0.0

		cout << "weight " << i << " = " << weights[i] << endl;
	}
	cout << "" << endl;
}


void calc_hidden_layers(int x)
{
	sum_h1 = (training_data[x][0] * weights[0]) + (training_data[x][1] * weights[2]) + (bias * weights[4]);
	sum_h2 = (training_data[x][0] * weights[1]) + (training_data[x][1] * weights[3]) + (bias * weights[5]);
	h1 = sigmoid_function(sum_h1);
	h2 = sigmoid_function(sum_h2);
}

void calc_output_neuron()
{
	sum_output = (h1 * weights[6]) + (h2 * weights[7]) + (bias * weights[8]);
	output_neuron = sigmoid_function(sum_output);
}

void calc_error(int x)
{
	error[x] = output_neuron - anwser_data[x]; 
}

void calc_derivatives(int x)
{
	derivative_O1 = -error[x] * (exp(sum_output) / pow((1 + exp(sum_output)), 2));
	derivative_h1 = (exp(sum_h1) / pow((1 + exp(sum_h1)), 2)) * weights[6] * derivative_O1;
	derivative_h2 = (exp(sum_h2) / pow((1 + exp(sum_h2)), 2)) * weights[7] * derivative_O1;
}

void calc_gradient(int x)
{
	gradients[0] = sigmoid_function(training_data[x][0]) * derivative_h1;
	gradients[1] = sigmoid_function(training_data[x][0]) * derivative_h2;
	gradients[2] = sigmoid_function(training_data[x][1]) * derivative_h1;
	gradients[3] = sigmoid_function(training_data[x][1]) * derivative_h2;
	gradients[4] = sigmoid_function(bias) * derivative_h1;
	gradients[5] = sigmoid_function(bias) * derivative_h2;
	gradients[6] = h1 * derivative_O1;
	gradients[7] = h2 * derivative_O1;
	gradients[8] = sigmoid_function(bias) * derivative_O1;
}

void calc_updates()
{
	for (int i = 0; i < 9; i++)
	{
		update_weights[i] = (learning_rate * gradients[i]) + (momentum * prev_weight_update[i]);
		prev_weight_update[i] = update_weights[i];
	}
}

void update_new_weights()
{
	for (int i = 0; i < 9; i++)
	{
		weights[i] = weights[i] + update_weights[i];
	}
}

void calc_RMSE_ERROR()
{
	RMSE_ERROR = sqrt((pow(error[0], 2) + pow(error[1], 2) + pow(error[2], 2) + pow(error[3], 2) / 4));
	cout << "RMSE error: " << RMSE_ERROR << endl;
	cout << "" << endl;
}
