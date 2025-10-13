package ai.nlp.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import okhttp3.*;

import java.io.IOException;
import java.io.InputStream;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

public class EmbeddingClient {
  private final String url;
  private final OkHttpClient http;
  private final ObjectMapper mapper = new ObjectMapper();

  public EmbeddingClient(String url) {
    this.url = url;
    this.http = new OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(120, TimeUnit.SECONDS)
        .build();
  }

  public float[] embedE5(String text, boolean isQuery) throws IOException {
    String prefixed = (isQuery ? "query: " : "passage: ") + text;

    RequestBody body = RequestBody.create(
        mapper.writeValueAsBytes(Map.of("input", List.of(prefixed))),
        MediaType.parse("application/json"));

    Request req = new Request.Builder()
        .url(url)                       // ví dụ http://localhost:8080/embeddings
        .post(body)
        .header("Accept", "application/json")
        .build();

    try (Response resp = http.newCall(req).execute()) {
      if (!resp.isSuccessful()) {
        // đọc nội dung lỗi (nếu có) 1 LẦN và đưa vào thông báo
        String errPayload = safeBodyString(resp.body());
        throw new IOException("TEI HTTP " + resp.code() + " - " + resp.message()
            + (errPayload == null ? "" : " | body: " + truncate(errPayload, 500)));
      }

      ResponseBody rb = resp.body();
      if (rb == null) throw new IOException("Empty response body from TEI.");

      // ĐỌC MỘT LẦN bằng stream -> parse JSON
      try (InputStream is = rb.byteStream()) {
        JsonNode root = mapper.readTree(is);

        JsonNode embNode = root.get("embeddings");
        if (embNode != null && embNode.isArray() && embNode.size() > 0) {
          return toFloatArray(embNode.get(0));
        }

        JsonNode data = root.get("data");
        if (data != null && data.isArray() && data.size() > 0) {
          JsonNode e = data.get(0).get("embedding");
          if (e != null && e.isArray()) return toFloatArray(e);
        }

        throw new IOException("Unexpected TEI response format: " + truncate(root.toString(), 500));
      }
    }
  }

  // ---- helpers ----
  private static String safeBodyString(ResponseBody b) {
    if (b == null) return null;
    try {
      return b.string();
    } catch (Exception e) {
      return null;
    }
  }

  private static String truncate(String s, int max) {
    return (s == null || s.length() <= max) ? s : s.substring(0, max) + "...";
  }

  private static float[] toFloatArray(JsonNode arr) {
    int n = arr.size();
    float[] out = new float[n];
    for (int i = 0; i < n; i++) out[i] = (float) arr.get(i).asDouble();
    return out;
  }
}


