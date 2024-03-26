package synapticgo

type NeuralNetwork struct {
	denses []*Dense
}

func NewNeuralNetwork(denses ...*Dense) *NeuralNetwork {
	return &NeuralNetwork{
		denses: denses,
	}
}

func (nn *NeuralNetwork) NeuralNetwork(denses ...*Dense) *NeuralNetwork {
	return NewNeuralNetwork(denses...)
}
