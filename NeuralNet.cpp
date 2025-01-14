#include <iostream>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <cassert>
#include <fstream>
#include <sstream>
#include <string>

using namespace std;

// Function prototypes
float ReLU(float x);
float computeLoss(const vector<float>& output, const vector<float>& target);
vector<float> computeLossDerivative(const vector<float>& output, const vector<float>& target);
void updateBiasesOutputLayer(vector<float>& b, const vector<float>& dL_dO, float learningRate);
void updateWeightsOutputLayer(const vector<float>& aHidden, vector<vector<float>>& w, const vector<float>& dL_dO, float learningRate);
float ReLU_derivative(float x);
void updateBiasesHiddenLayer(vector<float>& b, const vector<float>& dl_dh, float learningRate);
void updateWeightsHiddenLayer(const vector<float>& aPrev, vector<vector<float>>& w, const vector<float>& dl_dh, float learningRate);
void processLayer(const vector<float>& inputArr, const vector<vector<float>>& weightsArr, const vector<float>& biasArr, vector<float>& resultArr, int inputSize, int outputSize);
bool readSingleRow(std::ifstream& file, std::vector<float>& inputVector, std::vector<int>& labelVector);

int main() {
    srand(time(NULL));

    // Define network structure
    const int inputSize = 784;
    const int hiddenLayerSize = 50;
    const int outputSize = 10;
    const float learningRate = 0.6;

    // Initialize biases and weights
    vector<float> b0(hiddenLayerSize, 0.01f);  // Initialize biases with small positive value
    vector<float> b1(hiddenLayerSize, 0.01f);  // Initialize biases with small positive value
    vector<float> b2(outputSize, 0.01f);       // Initialize biases with small positive value

    vector<vector<float>> w0(inputSize, vector<float>(hiddenLayerSize));
    vector<vector<float>> w1(hiddenLayerSize, vector<float>(hiddenLayerSize));
    vector<vector<float>> w2(hiddenLayerSize, vector<float>(outputSize));

    // Initialize weights with random values between [-0.05, 0.05]
    for (int i = 0; i < inputSize; ++i) {
        for (int j = 0; j < hiddenLayerSize; ++j) {
            w0[i][j] = static_cast<float>(rand()) / RAND_MAX * 0.1f - 0.05f;  // Random value between -0.05 and 0.05
        }
    }
    for (int i = 0; i < hiddenLayerSize; ++i) {
        for (int j = 0; j < hiddenLayerSize; ++j) {
            w1[i][j] = static_cast<float>(rand()) / RAND_MAX * 0.1f - 0.05f;  // Random value between -0.05 and 0.05
        }
    }
    for (int i = 0; i < hiddenLayerSize; ++i) {
        for (int j = 0; j < outputSize; ++j) {
            w2[i][j] = static_cast<float>(rand()) / RAND_MAX * 0.1f - 0.05f;  // Random value between -0.05 and 0.05
        }
    }

    // Initialize activations and expected output
    
    vector<float> a1(hiddenLayerSize, 0.0f);
    vector<float> a2(hiddenLayerSize, 0.0f);
    vector<float> a3(outputSize, 0.0f);
    vector<float> a0; 
    vector<float> target;  


    std::string trainFile = "testc.csv";


    std::ifstream file(trainFile);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << trainFile << std::endl;
        return 1;
    }

    

    while (readSingleRow(file, a0, target)) {
        
            // Forward pass
            processLayer(a0, w0, b0, a1, inputSize, hiddenLayerSize);
            processLayer(a1, w1, b1, a2, hiddenLayerSize, hiddenLayerSize);
            processLayer(a2, w2, b2, a3, hiddenLayerSize, outputSize);

            // Print output activations and loss
            cout << "Output activations (a3): ";
            for (float val : a3) {
                cout << val << " ";
            }
            cout << endl;

            float loss = computeLoss(a3, target);
            cout << "LOSS: " << loss << endl;

            // Backward pass
            vector<float> lossDerivatives = computeLossDerivative(a3, target);

            // Output layer updates
            updateBiasesOutputLayer(b2, lossDerivatives, learningRate);
            updateWeightsOutputLayer(a2, w2, lossDerivatives, learningRate);

            // Hidden layer 2 backpropagation
            vector<float> dl_dh(hiddenLayerSize, 0.0f);
            for (int i = 0; i < hiddenLayerSize; ++i) {
                for (int j = 0; j < outputSize; ++j) {
                    dl_dh[i] += lossDerivatives[j] * w2[i][j];
                }
                dl_dh[i] *= ReLU_derivative(a2[i]);
            }
            updateBiasesHiddenLayer(b1, dl_dh, learningRate);
            updateWeightsHiddenLayer(a1, w1, dl_dh, learningRate);

            // Hidden layer 1 backpropagation
            vector<float> dl_dh2(hiddenLayerSize, 0.0f);
            for (int i = 0; i < hiddenLayerSize; ++i) {
                for (int j = 0; j < hiddenLayerSize; ++j) {
                    dl_dh2[i] += dl_dh[j] * w1[i][j];
                }
                dl_dh2[i] *= ReLU_derivative(a1[i]);
            }
            updateBiasesHiddenLayer(b0, dl_dh2, learningRate);
            updateWeightsHiddenLayer(a0, w0, dl_dh2, learningRate);
        }
    

    return 0;
}

// Function definitions
float ReLU(float x) {
    return x > 0 ? x : 0;
}

void processLayer(const vector<float>& inputArr, const vector<vector<float>>& weightsArr, const vector<float>& biasArr, vector<float>& resultArr, int inputSize, int outputSize) {
    assert(inputArr.size() == inputSize);
    assert(resultArr.size() == outputSize);
    assert(weightsArr.size() == inputSize);
    assert(weightsArr[0].size() == outputSize);

    for (int i = 0; i < outputSize; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < inputSize; ++j) {
            sum += inputArr[j] * weightsArr[j][i];
        }
        sum += biasArr[i];
        resultArr[i] = ReLU(sum);
    }
}

float computeLoss(const vector<float>& output, const vector<float>& target) {
    float loss = 0.0f;
    for (int i = 0; i < output.size(); ++i) {
        loss += (output[i] - target[i]) * (output[i] - target[i]);
    }
    return loss;
}

vector<float> computeLossDerivative(const vector<float>& output, const vector<float>& target) {
    vector<float> derivatives(output.size());
    for (int i = 0; i < output.size(); ++i) {
        derivatives[i] = 2 * (output[i] - target[i]);
    }
    return derivatives;
}

void updateBiasesOutputLayer(vector<float>& b, const vector<float>& dL_dO, float learningRate) {
    for (int j = 0; j < b.size(); ++j) {
        b[j] -= learningRate * dL_dO[j];
    }
}

void updateWeightsOutputLayer(const vector<float>& aHidden, vector<vector<float>>& w, const vector<float>& dL_dO, float learningRate) {
    for (int i = 0; i < aHidden.size(); ++i) {
        for (int j = 0; j < w[i].size(); ++j) {
            w[i][j] -= learningRate * dL_dO[j] * aHidden[i];
        }
    }
}

void updateWeightsHiddenLayer(const vector<float>& aPrev, vector<vector<float>>& w, const vector<float>& dl_dh, float learningRate) {
    for (int i = 0; i < aPrev.size(); ++i) {
        for (int j = 0; j < w[i].size(); ++j) {
            w[i][j] -= learningRate * dl_dh[j] * aPrev[i];
        }
    }
}

void updateBiasesHiddenLayer(vector<float>& b, const vector<float>& dl_dh, float learningRate) {
    for (int j = 0; j < b.size(); ++j) {
        b[j] -= learningRate * dl_dh[j];
    }
}

float ReLU_derivative(float x) {
    return x > 0 ? 1.0f : 0.0f;
}

bool readSingleRow(std::ifstream& file, std::vector<float>& inputVector, std::vector<float>& target) {
    std::string line;
    if (!std::getline(file, line)) {
        return false; // No more rows to read
    }

    if (line.empty()) {
        std::cerr << "Encountered empty line in the file." << std::endl;
        return false;
    }

    std::stringstream ss(line);
    std::string value;

    // Clear the vectors for the next row
    inputVector.clear();
    target.assign(10, 0.0f); // Initialize one-hot label vector with 10 zeros as floats

    // Read the label (first value in the row)
    if (!std::getline(ss, value, ',')) {
        std::cerr << "Missing label in the row." << std::endl;
        return false;
    }

    int label;
    try {
        label = std::stoi(value);
        if (label < 0 || label >= 10) {
            throw std::out_of_range("Label is out of valid range [0-9].");
        }
        target[label] = 1.0f; // Set the corresponding index to 1 for one-hot encoding
    } catch (const std::exception& e) {
        std::cerr << "Error parsing label: " << e.what() << std::endl;
        return false;
    }

    // Read the remaining 784 pixel values
    int pixelCount = 0;
    while (std::getline(ss, value, ',')) {
        try {
            float pixelValue = std::stof(value) / 255.0f; // Normalize the pixel value
            inputVector.push_back(pixelValue);
            ++pixelCount;
        } catch (const std::exception& e) {
            std::cerr << "Error parsing pixel value: " << e.what() << std::endl;
            return false;
        }
    }

    if (pixelCount != 784) {
        std::cerr << "Row does not contain exactly 784 pixel values. Found: " << pixelCount << std::endl;
        return false;
    }

    return true; // Successfully read a row
}
