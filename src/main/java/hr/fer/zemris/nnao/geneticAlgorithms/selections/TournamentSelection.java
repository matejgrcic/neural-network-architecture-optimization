package hr.fer.zemris.nnao.geneticAlgorithms.selections;

import hr.fer.zemris.nnao.geneticAlgorithms.Solution;

import java.util.*;

public class TournamentSelection implements Selection {

    private static final Random rand = new Random();

    private boolean allowDuplicates;
    private int numberOfCandindates;

    public TournamentSelection(int numberOfCandindates, boolean allowDuplicates) {
        this.allowDuplicates = allowDuplicates;
        this.numberOfCandindates = numberOfCandindates;
    }

    @Override
    public Solution[] selectParents(List<Solution> population) {
        List<Solution> selected = new ArrayList<>(numberOfCandindates);
        List<Solution> populationCopy = new ArrayList<>(population);
        while (selected.size() < numberOfCandindates) {
            Solution candidate = populationCopy.get(rand.nextInt(populationCopy.size()));
            if(!allowDuplicates){
                populationCopy.remove(candidate);
            }
            selected.add(candidate);
        }
        Collections.sort(selected, (o1, o2) -> (int) (o1.getFitness() - o2.getFitness()));
        return new Solution[]{selected.get(0), selected.get(1)};
    }
}
