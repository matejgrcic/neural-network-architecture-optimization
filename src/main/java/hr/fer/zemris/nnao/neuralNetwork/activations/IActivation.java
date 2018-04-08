package hr.fer.zemris.nnao.neuralNetwork.activations;

public interface IActivation {

    double calculateValue(double x);
    double calculateFirstDerivative(double x);
    String getStringRepresentation();
}
