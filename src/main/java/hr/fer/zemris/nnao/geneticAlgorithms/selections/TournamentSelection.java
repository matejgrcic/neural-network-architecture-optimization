package hr.fer.zemris.nnao.geneticAlgorithms.selections;

import hr.fer.zemris.nnao.geneticAlgorithms.Solution;

import java.util.*;

public class TournamentSelection implements Selection {

    private static final Random rand = new Random();

    private int numberOfCandindates;

    public TournamentSelection(int numberOfCandindates) {
        this.numberOfCandindates = numberOfCandindates;
    }

    @Override
    public Solution[] selectParents(List<Solution> population) {
        List<Solution> selected = new ArrayList<>(numberOfCandindates);
        while (selected.size() < numberOfCandindates) {
            selected.add(population.get(rand.nextInt(population.size())));
        }
        Collections.sort(selected, (o1, o2) -> (int) (o1.getFitness() - o2.getFitness()));
        return new Solution[]{selected.get(0), selected.get(1)};
    }
}
