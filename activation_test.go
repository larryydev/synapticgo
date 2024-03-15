package synapticgo_test

import (
	"math"
	"testing"

	"github.com/larryydev/synapticgo"
)

const tolerance = 1e-14

func TestSigmoid(t *testing.T) {
	cases := []struct {
		x, want float64
	}{
		{-5, 0.006692850924283},
		{-1, 0.268941421369995},
		{0, 0.5},
		{1, 0.731058578630004},
		{5, 0.993307149075716},
	}

	for _, c := range cases {
		got := synapticgo.Sigmoid(c.x)
		if math.Abs(got-c.want) > tolerance {
			t.Errorf("Sigmoid(%f) = %f, want %f", c.x, got, c.want)
		}
	}
}

func TestStepFunction(t *testing.T) {
	cases := []struct {
		x, want float64
	}{
		{-5, 0},
		{-1, 0},
		{0, 1},
		{1, 1},
		{5, 1},
	}

	for _, c := range cases {
		got := synapticgo.StepFunction(c.x)
		if got != c.want {
			t.Errorf("step(%f) = %f, want %f", c.x, got, c.want)
		}
	}
}

func TestTanh(t *testing.T) {
	cases := []struct {
		x, want float64
	}{
		{-5, -0.99990920426259},
		{-1, -0.76159415595576},
		{0, 0},
		{1, 0.761594155955764},
		{5, 0.999909204262595},
	}

	for _, c := range cases {
		got := synapticgo.Tanh(c.x)
		if math.Abs(got-c.want) > tolerance {
			t.Errorf("tanh(%f) = %f, want %f", c.x, got, c.want)
		}
	}
}

func TestRelu(t *testing.T) {
	cases := []struct {
		x, want float64
	}{
		{-5, 0},
		{-1, 0},
		{0, 0},
		{1, 1},
		{5, 5},
	}

	for _, c := range cases {
		got := synapticgo.Relu(c.x)
		if got != c.want {
			t.Errorf("relu(%f) = %f, want %f", c.x, got, c.want)
		}
	}
}

func TestLeakyRelu(t *testing.T) {
	cases := []struct {
		x, want float64
	}{
		{-5, -0.05},
		{-1, -0.01},
		{0, 0},
		{1, 1},
		{5, 5},
	}

	for _, c := range cases {
		got := synapticgo.LeakyRelu(c.x)
		if math.Abs(got-c.want) > tolerance {
			t.Errorf("leakyRelu(%f) = %f, want %f (diff = %f)", c.x, got, c.want, math.Abs(got-c.want))
		}
	}
}

func TestPRelu(t *testing.T) {
	cases := []struct {
		x, want float64
	}{
		{-5, -1.25},
		{-1, -0.25},
		{0, 0},
		{1, 1},
		{5, 5},
	}

	alpha := 0.25

	for _, c := range cases {
		got := synapticgo.PRelu(c.x, alpha)
		if math.Abs(got-c.want) > tolerance {
			t.Errorf("prelu(%f) = %f, want %f (diff = %f)", c.x, got, c.want, math.Abs(got-c.want))
		}
	}
}
