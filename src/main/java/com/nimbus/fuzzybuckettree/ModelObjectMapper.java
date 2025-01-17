package com.nimbus.fuzzybuckettree;

import com.fasterxml.jackson.annotation.JsonTypeInfo;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import com.nimbus.fuzzybuckettree.prediction.handlers.PredictionHandler;

public class ModelObjectMapper {

    public static final ObjectMapper MAPPER;

    static {
        MAPPER = new ObjectMapper().registerModule(new JavaTimeModule());

        // Add mixin to handle PredictionHandler type information
        @JsonTypeInfo(
                use = JsonTypeInfo.Id.CLASS,
                include = JsonTypeInfo.As.PROPERTY,
                property = "@class"
        )
        abstract class PredictionHandlerMixin {}

        MAPPER.addMixIn(PredictionHandler.class, PredictionHandlerMixin.class);
    }

    private ModelObjectMapper() {}
}