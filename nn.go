package synapticgo

type NeuralNetwork struct {
	denses []*Dense
}

func NewNeuralNetwork(denses ...*Dense) *NeuralNetwork {
	return &NeuralNetwork{
		denses: denses,
	}
}
