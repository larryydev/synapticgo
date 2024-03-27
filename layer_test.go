package synapticgo_test

import (
	"testing"

	nn "synapticgo"
)

func TestNewLayer(t *testing.T) {
	t.Run("creates a layer with correct input and weight sizes", func(t *testing.T) {
		n := 5
		layer := nn.NewLayer(n, true)
		layerInputs := layer.GetInputs()
		layerWeights := layer.GetWeights()

		if len(layerInputs) != n {
			t.Errorf("expected inputs to have length %d, but got %d", n, len(layerInputs))
		}

		if len(layerWeights) != n {
			t.Errorf("expected weights to have length %d, but got %d", n, len(layerWeights))
		}
	})
}

func TestCalculateSum(t *testing.T) {
	t.Run("calculates dot product correctly", func(t *testing.T) {
		layer := nn.NewEmptyLayer()

		layer.SetInputs([]float64{1.0, 2.0, 3.0})
		layer.SetWeights([]float64{4.0, 5.0, 6.0})
		layer.SetBias(0.0)

		expected := 1.0*4.0 + 2.0*5.0 + 3.0*6.0
		actual := layer.CalculateSum()

		if actual != expected {
			t.Errorf("expected dot product to be %f, but got %f", expected, actual)
		}
	})
}
