package synapticgo

import (
	"encoding/gob"
	"fmt"
	"math/rand"
	"os"
	"reflect"
	"runtime"
	"strings"
)

// Structures

type NodeStruct struct {
	value      float64
	weight     float64
	activation func(float64) float64
}

type LayerStruct struct {
	nodes []*NodeStruct
}

type NNStruct struct {
	layers []*LayerStruct
}

// Objects

func Node(args ...interface{}) *NodeStruct {
	var value, weight float64
	var activation func(float64) float64

	switch len(args) {
	case 0:
		// if no arguments are provided
		value = rand.Float64()
		weight = rand.Float64()
		activation = Sigmoid
	case 2:
		value = args[0].(float64)
		weight = args[1].(float64)
		activation = Sigmoid
	case 3:
		value = args[0].(float64)
		weight = args[1].(float64)
		activation = args[2].(func(float64) float64)
	default:
		panic("Invalid number of arguments for Node function")
	}

	return &NodeStruct{value: value, weight: weight, activation: activation}
}

func Layer(nodes ...*NodeStruct) *LayerStruct {
	return &LayerStruct{nodes: nodes}
}

func NN(layers ...*LayerStruct) *NNStruct {
	return &NNStruct{layers: layers}
}

// Public functions

func (nn *NNStruct) Forward(inputs []float64) []float64 {
	numberOfLayers := len(nn.layers)
	outputs := make([]float64, len(nn.layers[numberOfLayers-1].nodes))

	for i, node := range nn.layers[0].nodes {
		node.value = inputs[i]
	}

	for l := 1; l < numberOfLayers; l++ {
		for _, node := range nn.layers[l].nodes {
			sum := 0.0
			for _, prevNode := range nn.layers[l-1].nodes {
				sum += prevNode.value * prevNode.weight
			}
			node.value = node.activation(sum)
		}
	}

	for i, node := range nn.layers[numberOfLayers-1].nodes {
		outputs[i] = node.value
	}

	return outputs
}

func (nn *NNStruct) Backward(output []float64, target []float64, learningRate float64) {
	errors := make([]float64, len(output))
	for i := range output {
		errors[i] = target[i] - output[i]
	}

	for layerIndex := len(nn.layers) - 1; layerIndex >= 0; layerIndex-- {
		layer := nn.layers[layerIndex]
		var nextErrors []float64

		if layerIndex == len(nn.layers)-1 {
			for i, node := range layer.nodes {
				node.weight += learningRate * errors[i] * node.value * (1 - node.value)
			}
		} else {
			nextLayer := nn.layers[layerIndex+1]
			nextErrors = make([]float64, len(layer.nodes))

			for i, node := range layer.nodes {
				sum := 0.0
				for j, nextNode := range nextLayer.nodes {
					sum += nextNode.weight * errors[j] * nextNode.value * (1 - nextNode.value)
				}
				nextErrors[i] = sum
				node.weight += learningRate * sum * node.value * (1 - node.value)
			}
		}

		errors = nextErrors
	}
}

func (nn *NNStruct) Train(inputs [][]float64, targets [][]float64, epochs int, learningRate float64) {
	for epoch := 0; epoch < epochs; epoch++ {
		for i, input := range inputs {
			output := nn.Forward(input)
			nn.Backward(output, targets[i], learningRate)
		}
	}
}

func (nn *NNStruct) Save(filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)
	err = encoder.Encode(nn)
	if err != nil {
		return err
	}

	return nil
}

func (nn *NNStruct) VisualizeNN() {
	fmt.Println("Neural Network Structure:")
	for i, layer := range nn.layers {
		fmt.Printf("Layer %d:\n", i+1)
		for _, node := range layer.nodes {
			fmt.Printf("  Node: Value=%.2f, Weight=%.2f, Activation=%v\n", node.value, node.weight, getFunctionName(node.activation))
		}
	}
}

// Helper functions

func getFunctionName(i interface{}) string {
	fullString := runtime.FuncForPC(reflect.ValueOf(i).Pointer()).Name()
	lastIndex := strings.LastIndex(fullString, ".")
	return fullString[lastIndex+1:]
}
