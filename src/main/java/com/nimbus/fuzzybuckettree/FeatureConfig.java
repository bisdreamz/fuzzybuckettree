package com.nimbus.fuzzybuckettree;

import java.util.Arrays;

/**
 * Represents an established configuration of bucket sizing for a particular feature.
 * @param label The consistent label this bucket config applies to.
 * @param valueCount The number of values this feature houses. Count of value(s) provided
 *                   by this feature must always be consistent. Value order is always preserved.
 *                   If bucketing is enabled, bucket config length must equal value count.
 * @param buckets The optional array of bucket deltas, where each index in the buckets maps directly
 *                to the index of the values this feature returns. E.g. if bucketing is enabled
 *                and this feature returns a time series of 5 values, then there are 5 ordered
 *                bucket values in which each time series index value may have its own unique
 *                bucket width applied. If null or zero length, then no feature value aggregation
 *                is applied and feature values will be treated as exact match.
 */
public record FeatureConfig (String label, FeatureValueType type, int valueCount, float[] buckets) {

    public FeatureConfig (String label, FeatureValueType type, int valueCount, float[] buckets) {
        this.label = label;
        this.type = type;
        this.valueCount = valueCount;
        this.buckets = buckets;

        if (label == null || label.isEmpty())
            throw new IllegalArgumentException("Feature label cannot be null or empty");

        if (type == null)
            throw new IllegalArgumentException("Feature type cannot be null");

        if (!type.supportsBucketing() && buckets != null && buckets.length > 0)
            throw new IllegalArgumentException("Buckets enabled for feature but feature type doesnt support bucketing");

        if (buckets != null && buckets.length > 0 && buckets.length != valueCount)
            throw new IllegalArgumentException("Feature buckets length must equal value count when bucketing enabled for feature");
    }

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;

        if (o == null || !(o instanceof FeatureConfig))
            return false;

        FeatureConfig that = (FeatureConfig) o;

        return (this.label.equals(that.label)
                && this.valueCount == that.valueCount
                && Arrays.hashCode(this.buckets) == Arrays.hashCode(that.buckets));
    }

}
