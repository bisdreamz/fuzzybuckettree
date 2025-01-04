package com.nimbus.fuzzybuckettree.tuner;

import java.util.List;
import java.util.Map;

/**
 * Producer of bucket option combinations to explore for features, which
 * is used during the auto tuning process.
 */
public interface BucketIterator {

    /**
     * @return List of bucket option permutations to explore which
     * adhere to the behavior and restrictions provided to the
     * implementing bucket iterator. Is not guaranteed to be thread
     * safe.
     */
    public List<Map<String, float[]>> getNextBatch();

    public boolean hasMoreBatches();

    /**
     * @return Total number of resulting combinations for the given feature(s) value(s), bucket options,
     * and maxCombinations settings per feature to adhere to.
     */
    public long getTotalCombinations();

    /**
     * Reset iteratr index to the beginning
     */
    public void reset();

    public int getTotalBatches();

}
