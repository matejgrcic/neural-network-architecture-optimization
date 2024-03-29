package hr.fer.zemris.nnao.neuralNetwork;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

import java.util.Random;

public class NNUtil {

    public static Random random = new Random();

    public static RealMatrix[] createWeightMatrices(int[] architecture) {
        RealMatrix[] weights = new RealMatrix[architecture.length - 1];
        for (int i = 0; i < architecture.length - 1; ++i) {
            weights[i] = new Array2DRowRealMatrix(architecture[i] + 1, architecture[i + 1]);
        }

        for (RealMatrix matrix : weights) {
            int cnt = 0;
            double[] weightsArray = xavierInitialization(matrix.getRowDimension(),matrix.getColumnDimension());
            for (int i = 0; i < matrix.getRowDimension(); ++i) {
                for (int j = 0; j < matrix.getColumnDimension(); ++j) {
                    matrix.setEntry(i, j, weightsArray[cnt++]);
                }
            }
        }
        return weights;
    }

    public static double[] getWeights(int weightsNumber, RealMatrix[] weights) {
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

    public static int calculateNumberOfWeights(int[] architecture) {
        int number = 0;
        for (int i = 0; i < architecture.length - 1; ++i) {
            number += (architecture[i] + 1) * architecture[i + 1];
        }
        return number;
    }

    public static double[] createRandomArray(int arraySize) {
        double[] array = new double[arraySize];
        for (int i = 0; i < arraySize; ++i) {
            array[i] = random.nextDouble();
        }

        return array;
    }

    public static double[] xavierInitialization(int rows, int cols) {
        double[] weights = new double[rows*cols];
        int cnt = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                weights[cnt++] = random.nextGaussian() * (2.0 / (rows + cols - 1));
            }
        }
        return weights;
    }
}
