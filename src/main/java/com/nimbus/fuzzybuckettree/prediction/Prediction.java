package com.nimbus.fuzzybuckettree.prediction;

public class Prediction<T> {

    private final T prediction;
    private final float confidence;

    public Prediction(T prediction, float confidence) {
        this.prediction = prediction;
        this.confidence = confidence;

        if (confidence < 0.0f || confidence > 1.0f)
            throw new IllegalArgumentException("Confidence must be between 0.0f and 1.0f");
    }

    public float getConfidence() {
        return confidence;
    }

    public T getPrediction() {
        return prediction;
    }
}
