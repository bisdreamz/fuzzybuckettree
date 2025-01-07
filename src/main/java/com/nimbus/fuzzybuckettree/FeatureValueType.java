package com.nimbus.fuzzybuckettree;

public enum FeatureValueType {

    FLOAT(Float[].class, true),
    STRING(String[].class, false),
    INTEGER(Integer[].class, false),
    DOUBLE(Double[].class, true);

    private final Class<?> arrayClass;
    private final boolean supportsBucketing;

    FeatureValueType(Class<?> arrayClass, boolean supportsBucketing) {
        this.arrayClass = arrayClass;
        this.supportsBucketing = supportsBucketing;
    }

    public boolean supportsBucketing() { return supportsBucketing; }

    public Class<?> getArrayClass() {
        return arrayClass;
    }

}