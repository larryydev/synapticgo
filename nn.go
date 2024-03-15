package gonn

type Node struct {
	value      float64
	weight     float64
	activation string
}

type NN struct {
	layers [][]*Node
}

func NeuralNetwork(input int, numberOfNodeForEachHiddenLayer []int, output int) {

}

func newNode(value float64, weight float64, activation string) *Node {

}
