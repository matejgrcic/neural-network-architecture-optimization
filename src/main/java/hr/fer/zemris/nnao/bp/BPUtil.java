package hr.fer.zemris.nnao.bp;

import hr.fer.zemris.nnao.datasets.DatasetEntry;
import org.apache.commons.math3.linear.RealVector;

import java.util.ArrayList;
import java.util.List;

public class BPUtil {


    public static List<DatasetEntry>[] createBatches(int batchSize, List<DatasetEntry> dataset) {
        int batchNumber = dataset.size() % batchSize == 0 ? dataset.size() / batchSize : (dataset.size() / batchSize) + 1;
        List<DatasetEntry>[] batches = (List<DatasetEntry>[]) new List[batchNumber];

        int counter = 0;
        for (int i = 0; i < batches.length; ++i) {
            batches[i] = new ArrayList<>();
            for (int j = 0; j < batchSize; ++j) {
                batches[i].add(dataset.get(counter++));
                if (counter == dataset.size()) {
                    return batches;
                }
            }
        }
        return batches;
    }

    public static double calculateSetMSE(RealVector mseVector, int size) {
        double mse = 0;
        for (int j = 0; j < mseVector.getDimension(); ++j) {
            mse += mseVector.getEntry(j);
        }
        return mse / size;
    }

}
