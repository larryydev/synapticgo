package synapticgo_test

import (
	"math"
	"testing"

	nn "synapticgo"
)

func TestMeanSquaredError(t *testing.T) {
	tests := []struct {
		name  string
		yTrue float64
		yPred float64
		want  float64
	}{
		{"Perfect match", 1.0, 1.0, 0.0},
		{"Positive error", 1.0, 2.0, 1.0},
		{"Negative error", 1.0, 0.5, 0.25},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := nn.MeanSquaredError(tc.yTrue, tc.yPred)
			if math.Abs(got-tc.want) > 1e-6 {
				t.Errorf("MeanSquaredError(%f, %f) = %f, want %f", tc.yTrue, tc.yPred, got, tc.want)
			}
		})
	}
}

func TestMeanSquaredErrorPrime(t *testing.T) {
	tests := []struct {
		name  string
		yTrue float64
		yPred float64
		want  float64
	}{
		{"Perfect match", 1.0, 1.0, 0.0},
		{"Positive error", 1.0, 2.0, 2.0},
		{"Negative error", 1.0, 0.5, -1.0},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := nn.MeanSquaredErrorPrime(tc.yTrue, tc.yPred)
			if math.Abs(got-tc.want) > 1e-6 {
				t.Errorf("MeanSquaredErrorPrime(%f, %f) = %f, want %f", tc.yTrue, tc.yPred, got, tc.want)
			}
		})
	}
}

func TestBinaryCrossEntropy(t *testing.T) {
	tests := []struct {
		name  string
		yTrue float64
		yPred float64
		want  float64
	}{
		{"True positive", 1.0, 0.8, 0.22314355131420976},
		{"True negative", 0.0, 0.2, 0.22314355131420976},
		{"False positive", 0.0, 0.8, 1.6094379124341003},
		{"False negative", 1.0, 0.2, 1.6094379124341003},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := nn.BinaryCrossEntropy(tc.yTrue, tc.yPred)
			if math.Abs(got-tc.want) > 1e-6 {
				t.Errorf("BinaryCrossEntropy(%f, %f) = %f, want %f", tc.yTrue, tc.yPred, got, tc.want)
			}
		})
	}
}
