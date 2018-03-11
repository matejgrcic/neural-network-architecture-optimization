package hr.fer.zemris.nnao.neuralNetwork;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

import java.util.Random;

public class NNUtil {

    public static Random random = new Random();

    public static RealMatrix[] initializeWeights(int[] architecture) {
        RealMatrix[] weights = new RealMatrix[architecture.length - 1];
        for (int i = 0; i < architecture.length - 1; ++i) {
            weights[i] = new Array2DRowRealMatrix(architecture[i] + 1, architecture[i + 1]);
        }

        for (RealMatrix matrix : weights) {
            for (int i = 0; i < matrix.getRowDimension(); ++i) {
                for (int j = 0; j < matrix.getColumnDimension(); ++j) {
                    matrix.setEntry(i, j, random.nextDouble());
                }
            }
        }
        return weights;
    }
}
