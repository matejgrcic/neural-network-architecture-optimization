package hr.fer.zemris.nnao.observers.evaluators;

public class LoggerEvaluationObserver implements EvaluatorObserver {

    @Override
    public void update(double error) {
        System.out.println("\tChild error: " + error);
    }
}
