package synapticgo_test

import (
	"testing"

	nn "synapticgo"
)

func TestNewDense(t *testing.T) {
	t.Run("test dense sizes", func(t *testing.T) {
		inputsLen, weightLen := 2, 2
		layersLen := 3

		dense := nn.NewDense(inputsLen, layersLen, nil, true)
		layers := dense.GetLayers()

		if len(layers) != layersLen {
			t.Errorf("expected layers to be %d, but got %d", layersLen, len(layers))
		}

		for _, layer := range layers {
			inputs := layer.GetInputs()
			weights := layer.GetWeights()

			if len(inputs) != inputsLen {
				t.Errorf("expected inputs to be %d, but got %d", inputsLen, len(inputs))
			}

			if len(weights) != weightLen {
				t.Errorf("expected weights to be %d, but got %d", weightLen, len(weights))
			}
		}
	})
}
