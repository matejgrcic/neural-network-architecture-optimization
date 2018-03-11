package hr.fer.zemris.nnao.neuralNetwork.activations;

public class ActivationFunctions {

    public static IActivation Identity = new IActivation() {
        @Override
        public double calculateValue(double x) {
            return x;
        }

        @Override
        public double calculateFirstDerivative(double x) {
            return 1.;
        }
    };

    public static IActivation ReLU = new IActivation() {
        @Override
        public double calculateValue(double x) {
            return Math.max(0., x);
        }

        @Override
        public double calculateFirstDerivative(double x) {
            return x > 0. ? 1. : 0.;
        }
    };

    public static IActivation Sigmoid = new IActivation() {
        @Override
        public double calculateValue(double x) {
            return 1. / (1. + Math.exp(-x));
        }

        @Override
        public double calculateFirstDerivative(double x) {
            return calculateValue(x) * (1. - calculateValue(x));
        }
    };
}
