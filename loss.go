package synapticgo

import (
	"math"
)

func MSE(yTrue, yPred float64) float64 {
	return math.Pow(yTrue-yPred, 2)
}

func MSEPrime(yTrue, yPred float64) float64 {
	return 2 * (yPred - yTrue)
}

func BCE(yTrue, yPred float64) float64 {
	if yTrue == 1 {
		return -math.Log(yPred)
	}
	return -math.Log(1 - yPred)
}

func BCEPrime(yTrue, yPred float64) float64 {
	if yTrue == 1 {
		return -(1 - yPred) / yPred
	}
	return yPred / (1 - yPred)
}
