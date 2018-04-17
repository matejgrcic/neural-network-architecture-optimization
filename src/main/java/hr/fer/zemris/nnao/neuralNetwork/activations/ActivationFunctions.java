package hr.fer.zemris.nnao.neuralNetwork.activations;

public class ActivationFunctions {
    private static final String IdentityStringRepresentation = "Iden";
    private static final String ReLuStringRepresentation = "ReLU";
    private static final String SigmoidStringRepresentation = "Sigm";

    public static IActivation Identity = new IActivation() {
        @Override
        public double calculateValue(double x) {
            return x;
        }

        @Override
        public double calculateFirstDerivative(double x) {
            return 1.;
        }

        @Override
        public String getStringRepresentation() {
            return IdentityStringRepresentation;
        }

        @Override
        public String toString() {
            return "Identity";
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

        @Override
        public String getStringRepresentation() {
            return ReLuStringRepresentation;
        }

        @Override
        public String toString() {
            return "ReLU";
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

        @Override
        public String getStringRepresentation() {
            return SigmoidStringRepresentation;
        }

        @Override
        public String toString() {
            return "Sigmoid";
        }
    };

    public static IActivation[] allActivations = new IActivation[]{
            ActivationFunctions.Identity, ActivationFunctions.ReLU, ActivationFunctions.Sigmoid};


}
