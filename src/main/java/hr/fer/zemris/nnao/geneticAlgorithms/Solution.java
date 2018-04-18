package hr.fer.zemris.nnao.geneticAlgorithms;

import hr.fer.zemris.nnao.neuralNetwork.activations.IActivation;

import java.util.Arrays;
import java.util.Objects;

public class Solution {

    private IActivation[] activations;
    private int numberOfLayers;
    private int[] architecture;
    private double[] weights;
    private double fitness;

    public Solution(IActivation[] activations, int numberOfLayers, int[] architecture, double[] weights) {
        this.activations = activations;
        this.numberOfLayers = numberOfLayers;
        this.architecture = architecture;
        this.weights = weights;
    }

    public void setWeights(double[] weights) {
        this.weights = weights;
    }

    public IActivation[] getActivations() {
        return activations;
    }

    public int getNumberOfLayers() {
        return numberOfLayers;
    }

    public int[] getArchitecture() {
        return architecture;
    }

    public double[] getWeights() {
        return weights;
    }

    public double getFitness() {
        return fitness;
    }

    public void setFitness(double fitness) {
        this.fitness = fitness;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder(numberOfLayers+": ");
        for(int i = 0; i<numberOfLayers; ++i) {
            sb.append(architecture[i]+ " "+activations[i]+"/");
        }
        return sb.toString();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        Solution solution = (Solution) o;

        if (numberOfLayers != solution.numberOfLayers) return false;
        // Probably incorrect - comparing Object[] arrays with Arrays.equals
        if (!Arrays.equals(activations, solution.activations)) return false;
        if (!Arrays.equals(architecture, solution.architecture)) return false;
        return Arrays.equals(weights, solution.weights);
    }

    @Override
    public int hashCode() {
        int result = Arrays.hashCode(activations);
        result = 31 * result + numberOfLayers;
        result = 31 * result + Arrays.hashCode(architecture);
        return result;
    }
}
