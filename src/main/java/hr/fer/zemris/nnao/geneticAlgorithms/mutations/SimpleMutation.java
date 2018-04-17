package hr.fer.zemris.nnao.geneticAlgorithms.mutations;

import hr.fer.zemris.nnao.geneticAlgorithms.Solution;
import hr.fer.zemris.nnao.neuralNetwork.activations.ActivationFunctions;
import hr.fer.zemris.nnao.neuralNetwork.activations.IActivation;

import java.util.Random;

import static hr.fer.zemris.nnao.neuralNetwork.NNUtil.calculateNumberOfWeights;
import static hr.fer.zemris.nnao.neuralNetwork.NNUtil.createRandomArray;

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
        if (rand.nextDouble() > mutationProbability) {
            return solution;
        }

        Solution mutated = null;
        int index = rand.nextInt(solution.getNumberOfLayers() * 2);
        if (index % 2 == 0) {
            //mutiraj aktivacije
            index = index % solution.getNumberOfLayers();
            IActivation[] activations = solution.getActivations();
            activations[rand.nextInt(activations.length)] =
                    ActivationFunctions.allActivations[rand.nextInt(ActivationFunctions.allActivations.length)];
            mutated = new Solution(activations, solution.getNumberOfLayers(), solution.getArchitecture(), solution.getWeights());
        } else {
            //mutiraj arhitekt
            index = index % solution.getNumberOfLayers();
            int[] architecture = solution.getArchitecture();
            architecture[rand.nextInt(architecture.length-2)+1] = rand.nextInt(maxLayerSize - minLayerSize + 1) + minLayerSize;
            double[] weights = createRandomArray(calculateNumberOfWeights(architecture));
            mutated = new Solution(solution.getActivations(),solution.getNumberOfLayers(),architecture,weights);
        }

        return mutated;
    }
}
