package ai.nlp.service;

import okhttp3.*;
import com.fasterxml.jackson.databind.*;

import java.io.IOException;
import java.util.concurrent.TimeUnit;

public class OllamaClient implements LlmClient {
  private final String url;
  private final OkHttpClient http;
  private final ObjectMapper mapper = new ObjectMapper();
  private final String model;

  public OllamaClient(String url, String model) {
    this.url = url;
    this.model = model;
    this.http = new OkHttpClient.Builder()
        .connectTimeout(15, TimeUnit.SECONDS)
        .readTimeout(180, TimeUnit.SECONDS)
        .build();
  }

  @Override
  public String complete(String prompt) throws Exception {
    var payload = mapper.createObjectNode();
    payload.put("model", model);
    payload.put("prompt", prompt);
    payload.put("stream", false);
    payload.put("temperature", 0.2);
    payload.put("top_p", 0.8);
    payload.put("stream", false);

    RequestBody body = RequestBody.create(
        mapper.writeValueAsBytes(payload),
        MediaType.parse("application/json")
    );

    Request req = new Request.Builder()
        .url(url)
        .post(body)
        .build();

    try (Response resp = http.newCall(req).execute()) {
      if (!resp.isSuccessful()) {
        throw new IOException("Ollama error: " + resp.code() + " - " + resp.message());
      }

      var root = mapper.readTree(resp.body().byteStream());
      if (root.has("response"))
        return root.get("response").asText().trim();
      return root.toPrettyString();
    }
  }
}