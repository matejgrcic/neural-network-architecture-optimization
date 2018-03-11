package hr.fer.zemris.nnao.neuralNetwork;

import hr.fer.zemris.nnao.neuralNetwork.activations.IActivation;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import static hr.fer.zemris.nnao.neuralNetwork.NNUtil.initializeWeights;

public class NeuralNetwork implements INeuralNetwork{

    private RealMatrix[] weights;
    private IActivation[] activationFunctions;
    private int[] architecture;
    private int weightsNumber;

    public NeuralNetwork(int[] architecture, IActivation[] activationFunctions) {
        weights = initializeWeights(architecture);
        for (RealMatrix matrix : weights) {
            weightsNumber += matrix.getColumnDimension() * matrix.getRowDimension();
        }
        this.activationFunctions = activationFunctions;
        this.architecture = architecture;
    }

    public double[] forward(double[] input) {
        RealVector inputVector = new ArrayRealVector(input);
        inputVector = inputVector.append(1.);
        for (int i = 0; i < weights.length; ++i) {
            inputVector = weights[i].preMultiply(inputVector);
            if (i != weights.length - 1) {
                inputVector = inputVector.append(1.);
            }
        }
        return inputVector.toArray();
    }

    public IActivation[] getActivationFunctions() {
        return activationFunctions;
    }

    public void setActivationFunctions(IActivation[] activationFunctions) {
        this.activationFunctions = activationFunctions;
    }

    public int[] getNeuralNetworkArchitecture() {
        return architecture;
    }

    public double[] getWeights() {
        double[] weightsArray = new double[weightsNumber];
        int offset = 0;
        for (RealMatrix matrix : weights) {
            for (int i = 0; i < matrix.getRowDimension(); ++i) {
                for (int j = 0; j < matrix.getColumnDimension(); ++j) {
                    weightsArray[offset++] = matrix.getEntry(i, j);
                }
            }
        }
        return weightsArray;
    }

    public RealMatrix[] getWeightsMatrix() {
        return weights;
    }

    public int getWeightsNumber() {
        return weightsNumber;
    }

    public void setWeights(double[] weightsArray) {
        int offset = 0;
        for (RealMatrix matrix : weights) {
            for (int i = 0; i < matrix.getRowDimension(); ++i) {
                for (int j = 0; j < matrix.getColumnDimension(); ++j) {
                    matrix.setEntry(i, j, weightsArray[offset++]);
                }
            }
        }
    }

    public void setWeights(RealMatrix[] weights) {
        this.weights = weights;
    }
}
