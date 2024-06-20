package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"gonum.org/v1/gonum/mat"
)

// Activation function and its derivative (Sigmoid)
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func sigmoidDerivative(x float64) float64 {
	return x * (1.0 - x)
}

// NeuralNetwork structure
type NeuralNetwork struct {
	inputLayerSize      int
	hiddenLayerSize     int
	outputLayerSize     int
	weightsInputHidden  *mat.Dense
	weightsHiddenOutput *mat.Dense
}

// NewNeuralNetwork creates a new neural network with the given sizes
func NewNeuralNetwork(inputLayerSize, hiddenLayerSize, outputLayerSize int) *NeuralNetwork {
	rand.Seed(time.Now().UnixNano())

	weightsInputHidden := mat.NewDense(hiddenLayerSize, inputLayerSize, nil)
	weightsHiddenOutput := mat.NewDense(outputLayerSize, hiddenLayerSize, nil)

	for i := 0; i < hiddenLayerSize; i++ {
		for j := 0; j < inputLayerSize; j++ {
			weightsInputHidden.Set(i, j, rand.Float64())
		}
	}

	for i := 0; i < outputLayerSize; i++ {
		for j := 0; j < hiddenLayerSize; j++ {
			weightsHiddenOutput.Set(i, j, rand.Float64())
		}
	}

	return &NeuralNetwork{
		inputLayerSize:      inputLayerSize,
		hiddenLayerSize:     hiddenLayerSize,
		outputLayerSize:     outputLayerSize,
		weightsInputHidden:  weightsInputHidden,
		weightsHiddenOutput: weightsHiddenOutput,
	}
}

// Train the neural network
func (nn *NeuralNetwork) Train(inputs, targets *mat.Dense, epochs int, learningRate float64) {
	for epoch := 0; epoch < epochs; epoch++ {
		// Feedforward
		hiddenInput := mat.NewDense(0, 0, nil)
		hiddenInput.Mul(nn.weightsInputHidden, inputs)
		hiddenOutput := applyActivation(hiddenInput, sigmoid)

		finalInput := mat.NewDense(0, 0, nil)
		finalInput.Mul(nn.weightsHiddenOutput, hiddenOutput)
		finalOutput := applyActivation(finalInput, sigmoid)

		// Backpropagation
		outputErrors := mat.NewDense(0, 0, nil)
		outputErrors.Sub(targets, finalOutput)

		outputGradient := applyActivationDerivative(finalOutput, sigmoidDerivative)
		outputGradient.MulElem(outputGradient, outputErrors)
		outputGradient.Scale(learningRate, outputGradient)

		hiddenErrors := mat.NewDense(0, 0, nil)
		hiddenErrors.Mul(nn.weightsHiddenOutput.T(), outputErrors)

		hiddenGradient := applyActivationDerivative(hiddenOutput, sigmoidDerivative)
		hiddenGradient.MulElem(hiddenGradient, hiddenErrors)
		hiddenGradient.Scale(learningRate, hiddenGradient)

		// Update weights
		hiddenOutputT := hiddenOutput.T()
		deltaWeightsHO := mat.NewDense(0, 0, nil)
		deltaWeightsHO.Mul(outputGradient, hiddenOutputT)
		nn.weightsHiddenOutput.Add(nn.weightsHiddenOutput, deltaWeightsHO)

		inputsT := inputs.T()
		deltaWeightsIH := mat.NewDense(0, 0, nil)
		deltaWeightsIH.Mul(hiddenGradient, inputsT)
		nn.weightsInputHidden.Add(nn.weightsInputHidden, deltaWeightsIH)
	}
}

func applyActivation(m *mat.Dense, activationFunc func(float64) float64) *mat.Dense {
	r, c := m.Dims()
	result := mat.NewDense(r, c, nil)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			result.Set(i, j, activationFunc(m.At(i, j)))
		}
	}
	return result
}

func applyActivationDerivative(m *mat.Dense, activationDerivativeFunc func(float64) float64) *mat.Dense {
	r, c := m.Dims()
	result := mat.NewDense(r, c, nil)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			result.Set(i, j, activationDerivativeFunc(m.At(i, j)))
		}
	}
	return result
}

func main() {
	// Training data (XOR problem)
	inputs := mat.NewDense(4, 2, []float64{
		0, 0,
		0, 1,
		1, 0,
		1, 1,
	})

	targets := mat.NewDense(4, 1, []float64{
		0,
		1,
		1,
		0,
	})

	// Create neural network
	nn := NewNeuralNetwork(2, 2, 1)

	// Train the neural network
	nn.Train(inputs, targets, 10000, 0.1)

	// Test the neural network
	testInputs := mat.NewDense(4, 2, []float64{
		0, 0,
		0, 1,
		1, 0,
		1, 1,
	})

	hiddenInput := mat.NewDense(0, 0, nil)
	hiddenInput.Mul(nn.weightsInputHidden, testInputs)
	hiddenOutput := applyActivation(hiddenInput, sigmoid)

	finalInput := mat.NewDense(0, 0, nil)
	finalInput.Mul(nn.weightsHiddenOutput, hiddenOutput)
	finalOutput := applyActivation(finalInput, sigmoid)

	fmt.Println("Predictions:")
	for i := 0; i < 4; i++ {
		fmt.Printf("Input: %v, Output: %v\n", testInputs.RawRowView(i), finalOutput.RawRowView(i))
	}
}
