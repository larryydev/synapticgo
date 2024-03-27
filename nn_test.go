package synapticgo_test

import (
	"testing"

	nn "synapticgo"
)

func TestNewNeuralNetwork(t *testing.T) {
	t.Run("creates a EMPTY neural network", func(t *testing.T) {
		nn := nn.NewNeuralNetwork()

		denses := nn.GetDenses()

		if len(denses) != 0 {
			t.Errorf("expected layers to be %d, but got %d", 0, len(denses))
		}
	})

	t.Run("creates a EMPTY neural network", func(t *testing.T) {
		nn := nn.NewNeuralNetwork(
			nn.NewDense(5, 1, nn.Relu, false),
			nn.NewDense(2, 2, nn.Relu, false),
			nn.NewDense(2, 3, nn.Relu, false),
			nn.NewDense(2, 1, nn.Relu, false),
		)

		denses := nn.GetDenses()

		if len(denses) != 4 {
			t.Errorf("expected layers to be %d, but got %d", 4, len(denses))
		}
	})
}
