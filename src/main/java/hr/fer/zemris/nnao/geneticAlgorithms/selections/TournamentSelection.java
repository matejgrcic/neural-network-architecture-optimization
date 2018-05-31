package hr.fer.zemris.nnao.geneticAlgorithms.selections;

import hr.fer.zemris.nnao.geneticAlgorithms.Solution;

import java.util.*;

import static hr.fer.zemris.nnao.geneticAlgorithms.GAUtil.solutionComparator;

public class TournamentSelection implements Selection {

    private static final Random rand = new Random();

    private boolean allowDuplicates;
    private int numberOfCandidates;

    public TournamentSelection(int numberOfCandidates, boolean allowDuplicates) {
        this.allowDuplicates = allowDuplicates;
        this.numberOfCandidates = numberOfCandidates;
    }

    @Override
    public Solution[] selectParents(List<Solution> population) {
        List<Solution> selected = new ArrayList<>(numberOfCandidates);
        List<Solution> populationCopy = new ArrayList<>(population);
        while (selected.size() < numberOfCandidates) {
            Solution candidate = populationCopy.get(rand.nextInt(populationCopy.size()));
            if (!allowDuplicates) {
                populationCopy.remove(candidate);
            }
            selected.add(candidate);
        }
        Collections.sort(selected, solutionComparator);
        return new Solution[]{selected.get(0), selected.get(1)};
    }
}
