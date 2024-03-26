package synapticgo

import (
	"math/rand"
)

type Layer struct {
	inputs  []float64
	weights []float64
	bias    float64
}

func (l *Layer) DotProduct() float64 {
	dotProduct := 0.0

	for i, input := range l.inputs {
		dotProduct += input * l.weights[i]
	}

	return dotProduct + l.bias
}

func NewLayer(n int, useBias bool) *Layer {
	inputs := getFloats(n)
	weights := getFloats(n)

	bias := 0.0
	if useBias {
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
		res[i] = rand.Float64() * 100
	}
	return res
}
