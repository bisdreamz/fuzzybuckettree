package com.nimbus.fuzzybuckettree.tuner.reporters;

import java.util.Map;

public interface AccuracyReporter<T> {

    /**
     * Record the prediction accuracy with logic specific to this reporter
     * @param predicted Predicted value
     * @param actual Actual known value
     * @return true if prediction was considered corrct, false otherwise
     */
    public boolean record(T predicted, T actual);

    /**
     * @return The percentage of correct decision results,
     * expected 0f <> 1f
     */
    public float getTotalAccuracy();

    /**
     * A summary of accuracy by individual prediction outcome where possible.
     *      * Regression or related implementations may return a single mapped result
     *      * representing overall accuracy similar to {{@link #getTotalAccuracy()}}
     * @return Map of the actual correct outcome, to a {@link PredictionReport} detailing
     * the accuracy and frequency of each.
     */
    public Map<T, PredictionReport> getAccuracyReports();

    public AccuracyReporter<T> getNewInstance();

}
