package synapticgo

import (
	"fmt"
)

type Dense struct {
	layers     []*Layer
	activation func(float64) float64
}

func NewDense(shapeX int, shapeY int, activation func(float64) float64, useBias bool) *Dense {
	var layers []*Layer

	for i := 0; i < shapeY; i++ {
		newLayer := NewLayer(shapeX, useBias)
		layers = append(layers, newLayer)
	}

	if activation == nil {
		activation = Relu
	}

	return &Dense{
		layers:     layers,
		activation: activation,
	}
}

func (d *Dense) Dense(shapeX int, shapeY int, activation func(float64) float64, useBias bool) *Dense {
	return NewDense(shapeX, shapeY, activation, useBias)
}

func (d *Dense) Forward(inputs []float64) []float64 {
	if len(inputs) != len(d.layers[0].inputs) {
		panic(fmt.Sprintf("Invalid input shape: \n your input size: %d.\n nn's input size: %d", len(inputs), len(d.layers[0].inputs)))
	}

	d.layers[0].inputs = inputs

	for i := 0; i < len(d.layers)-1; i++ {
		curLayer := d.layers[i]
		nextLayer := d.layers[i+1]

		curLayerOutput := curLayer.DotProduct()
		outputValue := d.activation(curLayerOutput)

		for j := 0; j < len(nextLayer.inputs)-1; j++ {
			nextLayer.inputs[j] = outputValue
		}
	}

	return d.layers[len(d.layers)-1].inputs
}
