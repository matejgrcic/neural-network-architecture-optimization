package hr.fer.zemris.nnao.geneticAlgorithms.mutations;

import hr.fer.zemris.nnao.geneticAlgorithms.Solution;
import hr.fer.zemris.nnao.neuralNetwork.activations.ActivationFunctions;
import hr.fer.zemris.nnao.neuralNetwork.activations.IActivation;

import java.util.Random;

import static hr.fer.zemris.nnao.neuralNetwork.NNUtil.*;

public class SimpleMutation implements Mutation {

    private static final Random rand = new Random();

    private double mutationProbability;
    private int minLayerSize;
    private int maxLayerSize;

    public SimpleMutation(double mutationProbability, int minLayerSize, int maxLayerSize) {
        this.mutationProbability = mutationProbability;
        this.minLayerSize = minLayerSize;
        this.maxLayerSize = maxLayerSize;
    }

    @Override
    public Solution mutate(Solution solution) {
        // mutiraj slojeve
        for(int i = 0; i < solution.getNumberOfLayers()-2; ++i) {
            if (rand.nextDouble() > mutationProbability) {
                continue;
            }
            solution.getArchitecture()[i+1] = rand.nextInt(maxLayerSize - minLayerSize + 1) + minLayerSize;
        }
        //mutiraj arhitekturu
        for(int i = 0; i < solution.getNumberOfLayers(); ++i) {
            if (rand.nextDouble() > mutationProbability) {
                continue;
            }
            solution.getActivations()[i] = ActivationFunctions.allActivations[rand.nextInt(ActivationFunctions.allActivations.length)];
        }
        double[] weights = getWeights(calculateNumberOfWeights(solution.getArchitecture()),createWeightMatrices(solution.getArchitecture()));
        return new Solution(solution.getActivations(),solution.getNumberOfLayers(),solution.getArchitecture(), weights);
    }
}
