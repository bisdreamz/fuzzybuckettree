package com.nimbus.fuzzybuckettree;

public enum FeatureValueType {

    FLOAT(Float[].class, true),
    INTEGER(Integer[].class, true),
    DOUBLE(Double[].class, true),
    STRING(String[].class, false),
    BOOLEAN(Boolean[].class, false);

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