package synapticgo

import (
	"math/rand"
)

type Layer struct {
	inputs  []float64
	weights []float64
	bias    float64
}

func (l *Layer) GetInputs() []float64 {
	return l.inputs
}

func (l *Layer) SetInputs(inputs []float64) {
	l.inputs = inputs
}

func (l *Layer) GetWeights() []float64 {
	return l.weights
}

func (l *Layer) SetWeights(weights []float64) {
	l.weights = weights
}

func (l *Layer) GetBias() float64 {
	return l.bias
}

func (l *Layer) SetBias(bias float64) {
	l.bias = bias
}

func (l *Layer) DotProduct() float64 {
	dotProduct := 0.0

	for i, input := range l.inputs {
		dotProduct += input * l.weights[i]
	}

	return dotProduct + l.bias
}

func NewEmptyLayer() *Layer {
	return &Layer{}
}

func NewLayer(n int, usebias bool) *Layer {
	inputs := getFloats(n)
	weights := getFloats(n)

	bias := 0.0
	if usebias {
		bias = rand.Float64() * 1
	}

	return &Layer{
		inputs:  inputs,
		weights: weights,
		bias:    bias,
	}
}

func getFloats(n int) []float64 {
	res := make([]float64, n)
	for i := range res {
		res[i] = (rand.Float64()*2 - 1)
	}
	return res
}
