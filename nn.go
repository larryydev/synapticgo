package synapticgo

type NeuralNetwork struct {
	denses []*Dense
}

func (nn *NeuralNetwork) GetDenses() []*Dense {
	return nn.denses
}

func NewNeuralNetwork(denses ...*Dense) *NeuralNetwork {
	return &NeuralNetwork{
		denses: denses,
	}
}
