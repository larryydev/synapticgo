package gonn

import (
	"math"
)

func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Pow(math.E, -x))
}

func StepFunction(x float64) float64 {
	if x < 0 {
		return 0
	}
	return 1
}

func Tanh(x float64) float64 {
	numerator := (math.Pow(math.E, x) - math.Pow(math.E, -x))
	denominator := (math.Pow(math.E, x) + math.Pow(math.E, -x))
	return numerator / denominator
}

func Smht(x float64, a float64, b float64, c float64, d float64) float64 {
	numerator := (math.Pow(math.E, a*x) - math.Pow(math.E, -b*x))
	denominator := (math.Pow(math.E, c*x) + math.Pow(math.E, -d*x))
	return numerator / denominator
}

func Relu(x float64) float64 {
	if x <= 0 {
		return 0
	}
	return x
}

func Gelu(x float64) float64 {
	sig := 1 + math.Erf(x/math.Sqrt(2))
	return 0.5 * x * sig
}

func Softplus(x float64) float64 {
	return ln(1 + math.Pow(math.E, x))
}

func Elu(x float64, a float64) float64 {
	if x <= 0 {
		return a*math.Pow(math.E, x) - a
	}
	return x
}

func Selu(x float64) float64 {
	alpha, gamma := 1.6732632423543772, 1.0507009873554804
	if x < 0 {
		return gamma * (alpha*math.Pow(math.E, x) - alpha)
	}
	return gamma * x
}

func LeakyRelu(x float64) float64 {
	if x <= 0 {
		return 0.01 * x
	}
	return x
}

func PRelu(x float64, a float64) float64 {
	if x < 0 {
		return a * x
	}
	return x
}

func Silu(x float64) float64 {
	return x / (1 + math.Pow(math.E, -x))
}

func Gaussian(x float64) float64 {
	return math.Pow(math.E, math.Pow(-x, 2))
}

func f(x, a float64) float64 {
	return math.Exp(x) - a
}

func ln(n float64) float64 {
	if n <= 0 {
		return -1
	}

	if n == 1 {
		return 0
	}

	eps := 0.00001
	lo := 0.0
	hi := n

	for math.Abs(lo-hi) >= eps {
		m := float64((lo + hi) / 2.0)
		if f(m, n) < 0 {
			lo = m
		} else {
			hi = m
		}
	}

	return float64((lo + hi) / 2.0)
}
