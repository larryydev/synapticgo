package synapticgo

type NodeStruct struct {
	value      float64
	weight     float64
	activation func(float64) float64
}

type LayerStruct struct {
	nodes []*NodeStruct
}

type NNStruct struct {
	layers []*LayerStruct
}

func Node(value float64, weight float64, activation func(float64) float64) *NodeStruct {
	return &NodeStruct{value: value, weight: weight, activation: activation}
}

func Layer(nodes ...*NodeStruct) *LayerStruct {
	return &LayerStruct{nodes: nodes}
}

func NN(layers ...*LayerStruct) *NNStruct {
	return &NNStruct{layers: layers}
}

func (nn *NNStruct) Forward(inputs []float64) []float64 {
	numberOfLayers := len(nn.layers)
	outputs := make([]float64, len(nn.layers[numberOfLayers-1].nodes))

	for i, node := range nn.layers[0].nodes {
		node.value = inputs[i]
	}

	for l := 1; l < numberOfLayers; l++ {
		for _, node := range nn.layers[l].nodes {
			sum := 0.0
			for _, prevNode := range nn.layers[l-1].nodes {
				sum += prevNode.value * prevNode.weight
			}
			node.value = node.activation(sum)
		}
	}

	for i, node := range nn.layers[numberOfLayers-1].nodes {
		outputs[i] = node.value
	}

	return outputs
}
