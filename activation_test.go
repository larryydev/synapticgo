package synapticgo_test

import (
	"math"
	"testing"

	nn "synapticgo"
)

func TestSigmoid(t *testing.T) {
	cases := []struct {
		input    float64
		expected float64
	}{
		{0, 0.5},
		{-1, 1 / (1 + math.Exp(1))},
		{1, 1 / (1 + math.Exp(-1))},
		{-100, 1 / (1 + math.Exp(100))},
		{100, 1 / (1 + math.Exp(-100))},
	}

	for _, c := range cases {
		result := nn.Sigmoid(c.input)
		if math.Abs(result-c.expected) > 1e-10 {
			t.Errorf("Sigmoid(%f) = %f; expected %f", c.input, result, c.expected)
		}
	}
}

func TestTanh(t *testing.T) {
	cases := []struct {
		input    float64
		expected float64
	}{
		{0, 0},
		{-1, math.Tanh(-1)},
		{1, math.Tanh(1)},
		{-100, math.Tanh(-100)},
		{100, math.Tanh(100)},
	}

	for _, c := range cases {
		result := nn.Tanh(c.input)
		if math.Abs(result-c.expected) > 1e-10 {
			t.Errorf("Tanh(%f) = %f; expected %f", c.input, result, c.expected)
		}
	}
}

func TestRelu(t *testing.T) {
	cases := []struct {
		input    float64
		expected float64
	}{
		{0, 0},
		{-1, 0},
		{1, 1},
		{-100, 0},
		{100, 100},
	}

	for _, c := range cases {
		result := nn.Relu(c.input)
		if result != c.expected {
			t.Errorf("Relu(%f) = %f; expected %f", c.input, result, c.expected)
		}
	}
}

func TestLeakyRelu(t *testing.T) {
	cases := []struct {
		input    float64
		expected float64
	}{
		{0, 0},
		{-1, -0.01},
		{1, 1},
		{-100, -1},
		{100, 100},
	}

	for _, c := range cases {
		result := nn.LeakyRelu(c.input)
		if math.Abs(result-c.expected) > 1e-10 {
			t.Errorf("LeakyRelu(%f) = %f; expected %f", c.input, result, c.expected)
		}
	}
}
