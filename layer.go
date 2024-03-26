package synapticgo

import (
	"math/rand"
)

type Layer struct {
	inputs     []float64
	weights    []float64
	bias       float64
	activation func(float64) float64
}

func (l *Layer) Layer(n int, activation func(float64) float64, useBias bool) *Layer {
	return newLayer(n, activation, useBias)
}

func (l *Layer) DotProduct() float64 {
	dotProduct := 0.0

	for i, input := range l.inputs {
		dotProduct += input * l.weights[i]
	}

	return dotProduct + l.bias
}

func newLayer(n int, activation func(float64) float64, useBias bool) *Layer {
	inputs := getFloats(n)
	weights := getFloats(n)

	var activation_ func(float64) float64
	if activation == nil {
		activation_ = Sigmoid
	} else {
		activation_ = activation
	}

	bias := 0.0
	if useBias {
		bias = rand.Float64() * 1
	}

	return &Layer{
		inputs:     inputs,
		weights:    weights,
		bias:       bias,
		activation: activation_,
	}
}

func getFloats(n int) []float64 {
	res := make([]float64, n)
	for i := range res {
		res[i] = rand.Float64() * 100
	}
	return res
}
