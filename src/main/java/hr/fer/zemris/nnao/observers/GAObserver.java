package hr.fer.zemris.nnao.observers;

import hr.fer.zemris.nnao.geneticAlgorithms.AbstractGA;

public interface GAObserver {

    void update(AbstractGA geneticAlgorithm);
}
