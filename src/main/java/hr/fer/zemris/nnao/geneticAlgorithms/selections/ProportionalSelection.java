package hr.fer.zemris.nnao.geneticAlgorithms.selections;

import hr.fer.zemris.nnao.geneticAlgorithms.Solution;

import java.util.List;
import java.util.Random;

public class ProportionalSelection implements Selection {

    private static final Random rand = new Random();

    @Override
    public Solution[] selectParents(List<Solution> population) {

        double fitnessSum = 0;
        for (Solution solution : population) {
            fitnessSum += solution.getFitness();
        }

        Solution[] parents = new Solution[2];
        for (int i = 0; i < 2; ++i) {
            double currentTotalFitness = 0.;
            double desiredFitness = rand.nextDouble() * fitnessSum;
            for (int j = 0; j < population.size(); ++j) {
                currentTotalFitness += population.get(j).getFitness();
                if (currentTotalFitness > desiredFitness) {
                    parents[i] = population.get(j);
                    break;
                }
            }
            if (parents[i] == null) {
                parents[i] = population.get(population.size() - 1);
            }
        }
        return parents;
    }
}
