package test;

import hr.fer.zemris.nnao.datasets.DatasetEntry;
import hr.fer.zemris.nnao.datasets.DatasetUtils;

import java.io.IOException;
import java.util.List;

public class DatasetMain {

    public static void main(String[] args) throws IOException{
        List<DatasetEntry> data = DatasetUtils.createRastring2DDataset();
        System.out.println("Done");
    }
}
