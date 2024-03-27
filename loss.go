package synapticgo

import (
	"math"
)

func MeanSquaredError(yTrue, yPred float64) float64 {
	return math.Pow(yTrue-yPred, 2)
}

func MeanSquaredErrorPrime(yTrue, yPred float64) float64 {
	return 2 * (yPred - yTrue)
}

func BinaryCrossEntropy(yTrue, yPred float64) float64 {
	if yTrue == 1 {
		return -math.Log(yPred)
	}
	return -math.Log(1 - yPred)
}
