package synapticgo_test

import (
	"math"
	"testing"

	nn "github.com/larryydev/synapticgo"
)

func TestForward(t *testing.T) {
	layer1 := nn.Layer(
		nn.Node(0.0, 0.1, nn.Sigmoid),
		nn.Node(0.0, 0.2, nn.Sigmoid),
	)
	layer2 := nn.Layer(
		nn.Node(0.0, 0.3, nn.Sigmoid),
		nn.Node(0.0, 0.4, nn.Sigmoid),
		nn.Node(0.0, 0.5, nn.Sigmoid),
	)
	layer3 := nn.Layer(
		nn.Node(0.0, 0.6, nn.Sigmoid),
	)
	neuralNetwork := nn.NN(layer1, layer2, layer3)

	neuralNetwork.VisualizeNN()

	inputs := []float64{0.1, 0.2}
	expected := []float64{}
	outputs := neuralNetwork.Forward(inputs)
	assertFloatSlicesEqual(t, expected, outputs)
}

func assertFloatSlicesEqual(t *testing.T, expected, actual []float64) {
	if len(expected) != len(actual) {
		t.Errorf("Slices have different lengths: expected %d, got %d", len(expected), len(actual))
		return
	}

	for i := range expected {
		if math.Abs(expected[i]-actual[i]) > 1e-9 {
			t.Errorf("Values at index %d differ: expected %.9f, got %.9f", i, expected[i], actual[i])
		}
	}
}
