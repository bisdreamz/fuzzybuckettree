package com.nimbus.fuzzybuckettree;

import com.fasterxml.jackson.annotation.JsonProperty;


public record FeatureValuePair<V>(
        String label,
        V[] values,
        @JsonProperty("type") FeatureValueType type
) {
    public static FeatureValuePair<Float> ofFloat(String label, Float[] values) {
        return new FeatureValuePair<>(label, values, FeatureValueType.FLOAT);
    }

    public static FeatureValuePair<String> ofString(String label, String[] values) {
        return new FeatureValuePair<>(label, values, FeatureValueType.STRING);
    }

    public static FeatureValuePair<Integer> ofInteger(String label, Integer[] values) {
        return new FeatureValuePair<>(label, values, FeatureValueType.INTEGER);
    }

    public static FeatureValuePair<Double> ofDouble(String label, Double[] values) {
        return new FeatureValuePair<>(label, values, FeatureValueType.DOUBLE);
    }

    public static FeatureValuePair<Boolean> ofBoolean(String label, Boolean[] values) {
        return new FeatureValuePair<>(label, values, FeatureValueType.BOOLEAN);
    }
}