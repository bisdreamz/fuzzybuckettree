package com.nimbus.fuzzybuckettree.prediction;

public class NodePrediction<T> extends Prediction<T> {

    private final int depth;

    public NodePrediction(Prediction<T> prediction, int depth) {
        super(prediction.getPrediction(), prediction.getConfidence());
        this.depth = depth;

        if (depth < 0)
            throw new IllegalArgumentException("depth must be greater >= 0");
    }

    public int getDepth() {
        return depth;
    }

}
