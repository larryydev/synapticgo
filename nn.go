package synapticgo

import (
	"fmt"
	"math/rand"
	"reflect"
	"runtime"
	"strings"
)

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

func (nn *NNStruct) Train(inputs []float64, outputs []float64) {

}

func (nn *NNStruct) Save() {

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

func getFunctionName(i interface{}) string {
	fullString := runtime.FuncForPC(reflect.ValueOf(i).Pointer()).Name()
	lastIndex := strings.LastIndex(fullString, ".")
	return fullString[lastIndex+1:]
}
