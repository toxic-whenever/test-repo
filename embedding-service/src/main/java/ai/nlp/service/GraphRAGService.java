package ai.nlp.service;

import org.neo4j.driver.*;
import okhttp3.*;
import com.fasterxml.jackson.databind.*;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.io.IOException;
import java.util.*;
import java.util.concurrent.TimeUnit;

import static org.neo4j.driver.Values.parameters;

/**
 * GraphRAGService v3
 * - Retrieve top-k QA from Neo4j (vector index)
 * - Extract factual statements with confidence using Ollama
 * - Aggregate votes (confidence-weighted)
 * - Compose final answer using only reliable facts
 * <p>
 * Requirements:
 * - Neo4j running with vector index `qa_embedding_index`
 * - Text Embeddings Inference server at http://localhost:8080/embeddings
 * - Ollama running locally (e.g. mistral / llama3 / gemma2)
 */
public class GraphRAGService {

  private final Driver driver;
  private final EmbeddingClient embedClient;
  private final OllamaClient ollamaStrict;
  private final String dbName;
  private final String vectorIndexName = "qa_embedding_index";
  private final int topKVec = 30;
  private final int voteThreshold = 3;

  public GraphRAGService(Driver driver, EmbeddingClient embedClient,
                         String dbName, String ollamaUrl, String ollamaModel) {
    this.driver = driver;
    this.embedClient = embedClient;
    this.dbName = (dbName == null || dbName.isBlank()) ? "rag" : dbName;
    this.ollamaStrict = new OllamaClient(ollamaUrl, ollamaModel, 0.2, 0.9, false);
  }

  /**
   * Main RAG pipeline: ANN → Extract → Vote → Compose
   */
  public String answerPlain(String userQuery) throws Exception {
    float[] q = embedClient.embedE5(userQuery, true); // "query:"
    List<Double> qemb = toDoubleList(q);
    List<Cand> cands = vectorTopK(qemb, topKVec);

    if (cands.isEmpty()) {
      return "I'm not sure based on the available data.";
    }

    // Step 1. Extract facts from each answer
    List<Fact> allFacts = new ArrayList<>();
    for (Cand c : cands) {
      allFacts.addAll(extractFactsFromText(c.a, c.id));
    }

    // Step 2. Vote by confidence
    List<String> reliableFacts = selectReliableFacts(allFacts);
    if (reliableFacts.isEmpty()) {
      return "I'm not sure based on the available data.";
    }

    // Step 3. Compose one accurate sentence
    String finalText = composeOneSentence(reliableFacts);
    return cleanup(finalText);
  }

  // ---------------------- Neo4j retrieval ----------------------

  private List<Cand> vectorTopK(List<Double> qemb, int k) {
    List<Cand> out = new ArrayList<>();
    try (Session s = driver.session(SessionConfig.forDatabase(dbName))) {
      var rs = s.run("""
          WITH $qemb AS qemb
          CALL db.index.vector.queryNodes($index, $k, qemb)
          YIELD node, score
          RETURN node.id AS id, node.question AS q, node.answer AS a, score
          ORDER BY score DESC
          """, parameters("qemb", qemb, "k", k, "index", vectorIndexName));
      while (rs.hasNext()) {
        var r = rs.next();
        out.add(new Cand(
            r.get("id").asString(),
            r.get("q").asString(),
            r.get("a").asString(),
            r.get("score").asDouble()
        ));
      }
    }
    return out;
  }

  // ---------------------- Fact extraction ----------------------

  private List<Fact> extractFactsFromText(String answerText, String qaId) throws IOException {
    String prompt = """
        You are an information extractor.
        From the GIVEN TEXT, extract only factual statements that are explicitly written.
        For each fact, estimate its certainty level.
        
        Return JSON:
        {"facts":[{"text":"<fact>","certainty":"high|medium|low"}]}
        
        GIVEN TEXT:
        %s
        """.formatted(answerText);

    String resp = ollamaStrict.complete(prompt);
    ObjectMapper om = new ObjectMapper();
    List<Fact> facts = new ArrayList<>();

    try {
      JsonNode root = om.readTree(resp);
      for (JsonNode f : root.path("facts")) {
        String text = f.path("text").asText("");
        String cert = f.path("certainty").asText("medium");
        if (!text.isBlank()) {
          facts.add(new Fact(qaId, text, cert));
        }
      }
    } catch (Exception e) {
      // ignore malformed JSON
    }
    return facts;
  }

  private List<String> selectReliableFacts(List<Fact> allFacts) {
    Map<String, Integer> scoreMap = new HashMap<>();
    for (Fact f : allFacts) {
      int weight = switch (f.certainty) {
        case "high" -> 3;
        case "medium" -> 1;
        default -> 0;
      };
      String key = normalize(f.text);
      scoreMap.merge(key, weight, Integer::sum);
    }

    return scoreMap.entrySet().stream()
        .filter(e -> e.getValue() >= voteThreshold)
        .sorted((a, b) -> Integer.compare(b.getValue(), a.getValue()))
        .map(Map.Entry::getKey)
        .toList();
  }

  // ---------------------- Compose final answer ----------------------

  private String composeOneSentence(List<String> facts) throws IOException {
    String list = String.join(", ", facts);
    String prompt = """
        You are a precise summarizer.
        Write one clear and accurate English sentence using ONLY these facts:
        %s
        
        Do NOT add or infer anything not listed.
        Return plain text only.
        """.formatted(list);

    return ollamaStrict.complete(prompt).trim();
  }

  // ---------------------- Helpers ----------------------

  private static List<Double> toDoubleList(float[] v) {
    List<Double> L = new ArrayList<>(v.length);
    for (float x : v) L.add((double) x);
    return L;
  }

  private static String normalize(String s) {
    return s.trim().toLowerCase().replaceAll("\\s+", " ");
  }

  private static String cleanup(String s) {
    return s.replaceAll("\\[QA#.*?\\]", "")
        .replaceAll("\\s{2,}", " ")
        .trim();
  }

  // ---------------------- Data classes ----------------------

  static class Cand {
    final String id, q, a;
    final double score;

    Cand(String id, String q, String a, double score) {
      this.id = id;
      this.q = q;
      this.a = a;
      this.score = score;
    }
  }

  static class Fact {
    final String qaId;
    final String text;
    final String certainty;

    Fact(String qaId, String text, String certainty) {
      this.qaId = qaId;
      this.text = text;
      this.certainty = certainty;
    }
  }

  // ---------------------- Ollama Client ----------------------

  public static class OllamaClient {
    private final String url;
    private final String model;
    private final OkHttpClient http;
    private final ObjectMapper om = new ObjectMapper();
    private final double temperature;
    private final double topP;
    private final boolean stream;

    public OllamaClient(String url, String model, double temperature, double topP, boolean stream) {
      this.url = (url == null || url.isBlank()) ? "http://localhost:11434/api/generate" : url;
      this.model = (model == null || model.isBlank()) ? "mistral" : model;
      this.temperature = temperature;
      this.topP = topP;
      this.stream = stream;

      this.http = new OkHttpClient.Builder()
          .connectTimeout(15, TimeUnit.SECONDS)
          .readTimeout(180, TimeUnit.SECONDS)
          .build();
    }

    public String complete(String prompt) throws IOException {
      ObjectNode payload = om.createObjectNode();
      payload.put("model", model);
      payload.put("prompt", prompt);
      payload.put("temperature", temperature);
      payload.put("top_p", topP);
      payload.put("stream", stream);

      Request req = new Request.Builder()
          .url(url)
          .post(RequestBody.create(om.writeValueAsBytes(payload),
              MediaType.parse("application/json")))
          .build();

      try (Response resp = http.newCall(req).execute()) {
        if (!resp.isSuccessful()) {
          String err = resp.body() != null ? resp.body().string() : "";
          throw new IOException("Ollama HTTP " + resp.code() + " - " + err);
        }
        JsonNode root = om.readTree(resp.body().byteStream());
        if (root.has("response")) return root.get("response").asText();
        return root.toString();
      }
    }
  }

  // ---------------------- Demo main ----------------------

  public static void main(String[] args) throws Exception {
    String uri = System.getenv().getOrDefault("NEO4J_URI", "bolt://localhost:7687");
    String user = System.getenv().getOrDefault("NEO4J_USER", "neo4j");
    String pass = System.getenv().getOrDefault("NEO4J_PASS", "12345678");
    String db = System.getenv().getOrDefault("NEO4J_DB", "rag");
    String teiUrl = System.getenv().getOrDefault("TEI_URL", "http://localhost:8080/embeddings");
    String ollamaUrl = System.getenv().getOrDefault("OLLAMA_URL", "http://localhost:11434/api/generate");
    String ollamaModel = System.getenv().getOrDefault("OLLAMA_MODEL", "mistral");

    EmbeddingClient embed = new EmbeddingClient(teiUrl);

    try (Driver driver = GraphDatabase.driver(uri, AuthTokens.basic(user, pass))) {
      GraphRAGService srv = new GraphRAGService(driver, embed, db, ollamaUrl, ollamaModel);

      String question = (args.length > 0) ? String.join(" ", args)
          : "How do I pay for international roaming automatically?";

      String ans = srv.answerPlain(question);

      System.out.println("\n=== Question ===\n" + question);
      System.out.println("\n=== RAG Answer ===\n" + ans);
    }
  }
}