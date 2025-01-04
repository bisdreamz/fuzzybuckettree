package com.nimbus.fuzzybuckettree.tuner.reporters;

public class PredictionReport {

    private int samples;
    private float predictions;
    private float correct;

    /**
     * Increase predictions made with the associated prediction target weight, also
     * increments real sample count.
     * @param weight Weight of particular label being represented. Default is 1.
     * @param correct True if this prediction was correct or not
     */
    void increment(float weight, boolean correct) {
        if (weight <= 0.01 || weight > 100)
            throw new IllegalArgumentException("Weight must be between 0.01 and 100");

        samples++;

        predictions += weight;

        if (correct)
            this.correct += weight;
    }

    /**
     * @return Weighted prediction value
     */
    public int getPredictions() {
        return (int) predictions;
    }

    /**
     * @return Number of weighted correct predictions made
     */
    public int getCorrect() {
        return (int) correct;
    }

    /**
     * @return Number of actual samples seen during the test, without any weighting
     */
    public int getSamples() {
        return samples;
    }

    /**
     * @return Percentage of predictions made that were correct with
     * the provided class weighting
     */
    public float calcWeightedAccuracy() {
        if (predictions == 0 || correct == 0)
            return 0;

        return (float) correct / predictions;
    }
}
