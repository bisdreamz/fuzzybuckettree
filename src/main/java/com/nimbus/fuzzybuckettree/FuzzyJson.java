package com.nimbus.fuzzybuckettree;

import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.*;
import com.fasterxml.jackson.databind.node.TreeTraversingParser;

import java.io.IOException;

public class FuzzyJson {

    public static class FeatureValuePairSerializer extends JsonSerializer<FeatureValuePair<?>> {
        @Override
        public void serialize(FeatureValuePair<?> pair, JsonGenerator gen, SerializerProvider provider)
                throws IOException {
            gen.writeStartObject();
            gen.writeStringField("label", pair.label());
            gen.writeObjectField("values", pair.values());
            gen.writeStringField("type", pair.type().name());
            gen.writeEndObject();
        }
    }

    public static class FeatureValuePairDeserializer extends JsonDeserializer<FeatureValuePair<?>> {
        @Override
        public FeatureValuePair<?> deserialize(JsonParser p, DeserializationContext ctxt) throws IOException {
            JsonNode node = p.getCodec().readTree(p);
            ObjectMapper mapper = (ObjectMapper) p.getCodec();

            String label = node.get("label").asText();
            FeatureValueType type = FeatureValueType.valueOf(node.get("type").asText());
            JsonNode valuesNode = node.get("values");

            Object values = mapper.treeToValue(valuesNode, type.getArrayClass());

            switch (type) {
                case STRING:
                    return FeatureValuePair.ofString(label, (String[]) values);
                case FLOAT:
                    return FeatureValuePair.ofFloat(label, (Float[]) values);
                case INTEGER:
                    return FeatureValuePair.ofInteger(label, (Integer[]) values);
                case DOUBLE:
                    return FeatureValuePair.ofDouble(label, (Double[]) values);
                case BOOLEAN:
                    return FeatureValuePair.ofBoolean(label, (Boolean[]) values);
                default:
                    throw new IllegalArgumentException("Unsupported type: " + type);
            }
        }
    }

}