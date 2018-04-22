package hr.fer.zemris.nnao.geneticAlgorithms.evaluators;

import hr.fer.zemris.nnao.geneticAlgorithms.Solution;
import hr.fer.zemris.nnao.observers.evaluators.EvaluatorObserver;

import java.util.ArrayList;
import java.util.List;

public abstract class AbstractPopulationEvaluator implements PopulationEvaluator {

    protected List<EvaluatorObserver> observers = new ArrayList<>();


    public void addObserver(EvaluatorObserver observer) {
        observers.add(observer);
    }

    public void removeObserver(EvaluatorObserver observer) {
        observers.remove(observer);
    }

    protected void notifyObservers(double value) {
        new ArrayList<>(observers).forEach(t -> t.update(value));
    }

    public abstract double evaluateSolution(Solution solution);
}
