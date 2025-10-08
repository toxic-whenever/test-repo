package ai.graphrag;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import okhttp3.*;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.time.Duration;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class OllamaExtractor2 {

  /* ========================= CLI ========================= */
  record Args(Path input, Path output, String model, String baseUrl, int threads, int batchSize, int maxRetries,
              double temperature, int maxUnits, boolean keepQAPrefix, boolean useNerHints, int maxHints) {
  }

  static Args parseArgs(String[] argv) {
    Map<String, String> a = new HashMap<>();
    for (int i = 0; i < argv.length; i++) {
      if (argv[i].startsWith("--")) {
        String key = argv[i].substring(2);
        String val = (i + 1 < argv.length && !argv[i + 1].startsWith("--")) ? argv[++i] : "1";
        a.put(key, val);
      }
    }
    String path = "D:\\project\\embedding-service\\src\\main\\java\\ai\\nlp\\output\\";
    Path input = Path.of(a.getOrDefault("input", path + "nlp_output_test.jsonl"));
    Path output = Path.of(a.getOrDefault("output", path + "llm_structured_1.jsonl"));
    String model = a.getOrDefault("model", "qwen2.5:7b-instruct-q4_K_M");
    String base = a.getOrDefault("baseUrl", "http://localhost:11434");
    int threads = Integer.parseInt(a.getOrDefault("threads", "6"));
    int batch = Integer.parseInt(a.getOrDefault("batchSize", "8"));
    int retries = Integer.parseInt(a.getOrDefault("maxRetries", "3"));
    double temp = Double.parseDouble(a.getOrDefault("temperature", "0.0"));
    int maxUnits = Integer.parseInt(a.getOrDefault("maxUnits", "0"));
    boolean keepQAPrefix = Boolean.parseBoolean(a.getOrDefault("keepQAPrefix", "true"));
    boolean useNerHints = Boolean.parseBoolean(a.getOrDefault("useNerHints", "true"));
    int maxHints = Integer.parseInt(a.getOrDefault("maxHints", "30"));
    return new Args(input, output, model, base, threads, batch, retries, temp, maxUnits, keepQAPrefix, useNerHints,
        maxHints);
  }

  /* ========================= Model ========================= */
  static String md5hex16(String s) {
    try {
      MessageDigest md = MessageDigest.getInstance("MD5");
      byte[] d = md.digest(s.getBytes(StandardCharsets.UTF_8));
      StringBuilder sb = new StringBuilder();
      for (byte b : d) sb.append(String.format("%02x", b));
      return sb.substring(0, 16);
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  }

  static class Unit {
    String docId;
    String chunkId;
    String question;
    String answer;
    String type;
    String text;
    List<Hint> hints;
    JsonNode raw;
  }

  static class Hint {
    String name;
    String type;
  }

  static List<Unit> buildUnits(Path input, int maxUnits, boolean keepQAPrefix, boolean useNerHints, int maxHints) throws IOException {
    ObjectMapper om = new ObjectMapper();
    List<Unit> units = new ArrayList<>();
    try (BufferedReader br = Files.newBufferedReader(input, StandardCharsets.UTF_8)) {
      String line;
      int idx = 0;
      int added = 0;
      while ((line = br.readLine()) != null) {
        idx++;
        line = line.trim();
        if (line.isEmpty()) continue;
        JsonNode node;
        try {
          node = om.readTree(line);
        } catch (Exception ex) {
          continue;
        }

        String docId = null;
        if (node.has("rowIndex") && node.get("rowIndex").isNumber()) {
          docId = "ROW_" + node.get("rowIndex").asText();
        }
        if (docId == null) docId = firstText(node, List.of("doc_id", "document_id", "id", "source_id", "docId"));
        if (docId == null || docId.isBlank()) docId = "DOC_" + idx;

        String q = node.has("question") && node.get("question").isTextual() ? node.get("question").asText() : null;
        String a = node.has("answer") && node.get("answer").isTextual() ? node.get("answer").asText() : null;
        String t = node.has("type") && node.get("type").isTextual() ? node.get("type").asText() : null;

        String combined = null;
        if (q != null || a != null) {
          String qPart = (q != null ? (keepQAPrefix ? "Q: " : "") + q : (keepQAPrefix ? "Q:" : ""));
          String aPart = (a != null ? (keepQAPrefix ? "A: " : "") + a : (keepQAPrefix ? "A:" : ""));
          combined = (qPart + "\n" + aPart).trim();
        }
        if ((combined == null || combined.isBlank()) && node.has("sentences") && node.get("sentences").isArray()) {
          StringBuilder sb = new StringBuilder();
          for (JsonNode s : node.get("sentences")) {
            if (s.has("text") && s.get("text").isTextual()) {
              String st = s.get("text").asText().trim();
              if (!st.isBlank()) sb.append(st).append('\n');
            }
          }
          combined = sb.toString().trim();
        }
        if (combined == null || combined.isBlank()) {
          combined = rebuildFromTokens(node);
        }
        if (combined == null || combined.isBlank()) continue;

        List<Hint> hints = useNerHints ? collectNerHints(node, maxHints) : Collections.emptyList();

        String chunkId = md5hex16(docId + "||" + idx);
        Unit u = new Unit();
        u.docId = docId;
        u.chunkId = chunkId;
        u.question = q;
        u.answer = a;
        u.type = t;
        u.text = combined;
        u.hints = hints;
        u.raw = node;
        units.add(u);
        added++;
        if (maxUnits > 0 && added >= maxUnits) break;
      }
    }
    return units;
  }

  static String firstText(JsonNode node, List<String> keys) {
    for (String k : keys) {
      if (node.has(k) && node.get(k).isTextual()) {
        String v = node.get(k).asText();
        if (v != null && !v.isBlank()) return v;
      }
    }
    return null;
  }

  static String rebuildFromTokens(JsonNode node) {
    if (node.isObject() || node.isArray()) {
      Deque<JsonNode> dq = new ArrayDeque<>();
      dq.add(node);
      while (!dq.isEmpty()) {
        JsonNode cur = dq.removeFirst();
        if (cur.isObject() && cur.has("tokens") && cur.get("tokens").isArray()) {
          StringBuilder sb = new StringBuilder();
          for (JsonNode t : cur.get("tokens")) {
            if (t.has("word")) sb.append(t.get("word").asText()).append(' ');
          }
          String s = sb.toString().trim();
          if (!s.isBlank()) return s;
        }
        if (cur.isObject()) cur.fields().forEachRemaining(e -> dq.add(e.getValue()));
        else if (cur.isArray()) cur.forEach(dq::add);
      }
    }
    return null;
  }

  static List<Hint> collectNerHints(JsonNode node, int maxHints) {
    List<Hint> hints = new ArrayList<>();
    if (!(node.isObject() || node.isArray())) return hints;
    Deque<JsonNode> dq = new ArrayDeque<>();
    dq.add(node);
    Set<String> seen = new HashSet<>();
    while (!dq.isEmpty()) {
      JsonNode cur = dq.removeFirst();
      if (cur.isObject() && cur.has("tokens") && cur.get("tokens").isArray()) {
        List<String> buf = new ArrayList<>();
        String curType = null;
        for (JsonNode t : cur.get("tokens")) {
          String ner = t.has("ner") ? t.get("ner").asText("O") : "O";
          String w = t.has("word") ? t.get("word").asText("") : "";
          if (!"O".equals(ner) && !w.isBlank()) {
            if (curType != null && !ner.equals(curType)) {
              pushHint(hints, seen, String.join(" ", buf).trim(), curType);
              buf = new ArrayList<>();
            }
            buf.add(w);
            curType = ner;
          } else {
            if (!buf.isEmpty()) {
              pushHint(hints, seen, String.join(" ", buf).trim(), curType);
              buf = new ArrayList<>();
              curType = null;
            }
          }
          if (hints.size() >= maxHints) break;
        }
        if (!buf.isEmpty() && hints.size() < maxHints) pushHint(hints, seen, String.join(" ", buf).trim(), curType);
      }
      if (hints.size() >= maxHints) break;
      if (cur.isObject()) cur.fields().forEachRemaining(e -> dq.add(e.getValue()));
      else if (cur.isArray()) cur.forEach(dq::add);
    }
    return hints;
  }

  static void pushHint(List<Hint> hints, Set<String> seen, String name, String typ) {
    if (name == null || name.isBlank()) return;
    String key = (name + "|" + typ).toLowerCase(Locale.ROOT);
    if (seen.add(key)) {
      Hint h = new Hint();
      h.name = name;
      h.type = typ == null ? "Entity" : typ;
      hints.add(h);
    }
  }

  /* ========================= Ollama client ========================= */
  static final MediaType JSON = MediaType.parse("application/json; charset=utf-8");

  static class OllamaClient implements Closeable {
    final OkHttpClient http;
    final String baseUrl; // e.g., http://localhost:11434
    final String model;
    final double temperature;
    final ObjectMapper om = new ObjectMapper();

    OllamaClient(String baseUrl, String model, double temperature) {
      this.baseUrl = baseUrl.endsWith("/") ? baseUrl.substring(0, baseUrl.length() - 1) : baseUrl;
      this.model = model;
      this.temperature = temperature;
      this.http =
          new OkHttpClient.Builder().callTimeout(Duration.ofSeconds(120)).connectTimeout(Duration.ofSeconds(20)).readTimeout(Duration.ofSeconds(120)).build();
    }

    public Map<String, Object> chatJSON(String system, String user) throws IOException {
      Map<String, Object> payload = new LinkedHashMap<>();
      payload.put("model", model);
      payload.put("stream", false);
      Map<String, Object> opts = new LinkedHashMap<>();
      opts.put("temperature", temperature);
      payload.put("options", opts);
      List<Map<String, String>> messages = new ArrayList<>();
      messages.add(Map.of("role", "system", "content", system));
      messages.add(Map.of("role", "user", "content", user));
      payload.put("messages", messages);

      RequestBody body = RequestBody.create(om.writeValueAsBytes(payload), JSON);
      Request req = new Request.Builder().url(baseUrl + "/api/chat").post(body).build();
      try (Response resp = http.newCall(req).execute()) {
        if (!resp.isSuccessful()) throw new IOException("HTTP " + resp.code());
        String s = resp.body().string();
        return om.readValue(s, new TypeReference<Map<String, Object>>() {
        });
      }
    }

    @Override
    public void close() throws IOException { /* nothing */ }
  }

  /* ========================= Prompt (EN) ========================= */
  static final String SYSTEM_PROMPT = ("You are a precise knowledge extraction system. " + "From the given TEXT (may " +
      "include Question/Answer) and HINTS (from NER tokens), " + "return ONLY a valid JSON object following the " +
      "SCHEMA. No explanations.");

  static String buildUserPrompt(Unit u) {
    StringBuilder sb = new StringBuilder();
    sb.append("TEXT:\n").append(u.text).append("\n\n");
    if (u.hints != null && !u.hints.isEmpty()) {
      sb.append("HINTS (NER):\n");
      for (Hint h : u.hints)
        sb.append("- ").append(h.name).append(" :: ").append(h.type).append("\n");
      sb.append("\n");
    }
    if (u.type != null && !u.type.isBlank()) {
      sb.append("TYPE: ").append(u.type).append("\n");
    }
    sb.append("SCHEMA (return exactly this JSON shape):\n");
    sb.append("{\n\"entities\": [{\"name\": \"...\", \"type\": \"...\"}],\n");
    sb.append("  \"relations\": [{\"head\": \"...\", \"relation\": \"...\", \"tail\": \"...\", " + "\"confidence\": 0" +
        ".0}],\n");
    sb.append("  \"intent\": \"...\"\n}\n");
    return sb.toString();
  }

  /* ========================= Output ========================= */
  static class ResultLine {
    String doc_id;
    String chunk_id;
    String question;
    String answer;
    String type;
    String text;
    List<Map<String, Object>> entities = new ArrayList<>();
    List<Map<String, Object>> relations = new ArrayList<>();
    String intent = "";
    Object _llm_raw;
  }

  static void postProcess(ResultLine rl) {
    // 0) Guard
    if (rl == null) return;
    if (rl.entities == null) rl.entities = new ArrayList<>();
    if (rl.relations == null) rl.relations = new ArrayList<>();

    // 1) Split coordinated entities like "X or Y" / "X and Y"
    List<Map<String, Object>> expanded = new ArrayList<>();
    for (Map<String, Object> e : rl.entities) {
      String name = String.valueOf(e.getOrDefault("name", "")).trim();
      String typ = String.valueOf(e.getOrDefault("type", "Entity")).trim();
      if (name.isEmpty()) continue;
      String lower = name.toLowerCase(Locale.ROOT);
      if (lower.contains(" or ") || lower.contains(" and ")) {
        String[] parts = lower.contains(" or ") ? name.split("\bor\b") : name.split("\band\b");
        for (String p : parts) {
          String nn = p.trim();
          if (nn.isEmpty()) continue;
          expanded.add(Map.of("name", nn, "type", typ));
        }
      } else {
        expanded.add(Map.of("name", name, "type", typ));
      }
    }
    rl.entities = dedupEntities(expanded);

    // 2) Heuristics: promote PRODUCT->BENEFIT for allowance/credit/benefit names
    for (Map<String, Object> e : rl.entities) {
      String name = String.valueOf(e.getOrDefault("name", ""));
      String typ = String.valueOf(e.getOrDefault("type", ""));
      String lname = name.toLowerCase(Locale.ROOT);
      if ("PRODUCT".equalsIgnoreCase(typ)) {
        if (lname.contains("allowance") || lname.contains("benefit") || lname.contains("credit")) {
          e.put("type", "BENEFIT");
        }
      }
    }

    // 3) Ensure FEATURE entity 'auto-pay' if text mentions it
    if (rl.text != null && rl.text.toLowerCase(Locale.ROOT).contains("auto-pay")) {
      if (findEntity(rl.entities, "auto-pay") == null) {
        rl.entities.add(new HashMap<>(Map.of("name", "auto-pay", "type", "FEATURE")));
      }
    }
    // Dedup after adding
    rl.entities = dedupEntities(rl.entities);

    // 4) Fix relations: if accepts_method has head=APP while FEATURE 'auto-pay' exists → redirect
    boolean hasAutoPay = findEntity(rl.entities, "auto-pay") != null;
    String appName = firstEntityByType(rl.entities, "APP");
    if (hasAutoPay) {
      // Ensure supports edge from APP to auto-pay
      if (appName != null) {
        ensureRelation(rl.relations, appName, "supports", "auto-pay", 0.9);
      }
      // Redirect accepts_method edges
      for (Map<String, Object> r : rl.relations) {
        String rel = String.valueOf(r.getOrDefault("relation", ""));
        String head = String.valueOf(r.getOrDefault("head", ""));
        if ("accepts_method".equals(rel) && appName != null && appName.equals(head)) {
          r.put("head", "auto-pay");
          // boost confidence slightly if not set
          if (!(r.get("confidence") instanceof Number)) r.put("confidence", 0.85);
        }
      }
    }

    // 5) Clamp relation labels and fill default confidence
    for (Map<String, Object> r : rl.relations) {
      String rel = String.valueOf(r.getOrDefault("relation", ""));
      // Map common variants → allowed set
      String norm = switch (rel.toLowerCase(Locale.ROOT)) {
        case "include", "contains", "contain" -> "includes";
        case "support", "enable" -> rel.toLowerCase(Locale.ROOT).startsWith("enable") ? "enables" : "supports";
        case "use" -> "uses";
        default -> rel;
      };
      r.put("relation", norm);
      // Default confidence
      if (!(r.get("confidence") instanceof Number)) r.put("confidence", 0.75);
      try {
        double c = Double.parseDouble(String.valueOf(r.get("confidence")));
        if (c <= 0.0) r.put("confidence", 0.75);
        if (c > 1.0) r.put("confidence", 1.0);
      } catch (Exception ignore) {
        r.put("confidence", 0.75);
      }
    }

    // 6) Ensure head/tail exist in entities; drop invalid relations
    List<Map<String, Object>> ok = new ArrayList<>();
    for (Map<String, Object> r : rl.relations) {
      String h = String.valueOf(r.getOrDefault("head", ""));
      String t = String.valueOf(r.getOrDefault("tail", ""));
      if (findEntity(rl.entities, h) != null && findEntity(rl.entities, t) != null) ok.add(r);
    }
    rl.relations = ok;
  }

  static Map<String, Object> findEntity(List<Map<String, Object>> ents, String name) {
    if (name == null) return null;
    for (Map<String, Object> e : ents) {
      if (name.equalsIgnoreCase(String.valueOf(e.getOrDefault("name", "")))) return e;
    }
    return null;
  }

  static String firstEntityByType(List<Map<String, Object>> ents, String type) {
    for (Map<String, Object> e : ents) {
      if (type.equalsIgnoreCase(String.valueOf(e.getOrDefault("type", "")))) {
        return String.valueOf(e.getOrDefault("name", ""));
      }
    }
    return null;
  }

  static void ensureRelation(List<Map<String, Object>> rels, String head, String rel, String tail, double conf) {
    for (Map<String, Object> r : rels) {
      String h = String.valueOf(r.getOrDefault("head", ""));
      String re = String.valueOf(r.getOrDefault("relation", ""));
      String t = String.valueOf(r.getOrDefault("tail", ""));
      if (h.equalsIgnoreCase(head) && re.equalsIgnoreCase(rel) && t.equalsIgnoreCase(tail)) return;
    }
    rels.add(new HashMap<>(Map.of("head", head, "relation", rel, "tail", tail, "confidence", conf)));
  }

  static List<Map<String, Object>> dedupEntities(List<Map<String, Object>> in) {
    LinkedHashMap<String, Map<String, Object>> map = new LinkedHashMap<>();
    for (Map<String, Object> e : in) {
      String name = String.valueOf(e.getOrDefault("name", "")).trim();
      String typ = String.valueOf(e.getOrDefault("type", "Entity")).trim();
      if (name.isEmpty()) continue;
      String key = (name.toLowerCase(Locale.ROOT) + "|" + typ.toLowerCase(Locale.ROOT));
      map.putIfAbsent(key, new HashMap<>(Map.of("name", name, "type", typ)));
    }
    return new ArrayList<>(map.values());
  }

  static ResultLine normalizeLLM(Unit u, String content, ObjectMapper om) {
    ResultLine rl = new ResultLine();
    rl.doc_id = u.docId;
    rl.chunk_id = u.chunkId;
    rl.question = u.question;
    rl.answer = u.answer;
    rl.type = u.type;
    rl.text = u.text;
    String t = content.trim();
    t = t.replaceAll("^```json\s*", "");
    t = t.replaceAll("^```\s*", "");
    t = t.replaceAll("```$", "");
    t = t.replace('\'', '"');
    t = t.replaceAll(",\s*([}\\]])", "$1");
    boolean ok = false;
    try {
      JsonNode node = om.readTree(t);
      ok = fillFromNode(rl, node);
    } catch (Exception ignore) {
    }
    if (!ok) {
      int i = t.indexOf('{');
      int j = t.lastIndexOf('}');
      if (i >= 0 && j > i) {
        String frag = t.substring(i, j + 1);
        try {
          JsonNode node = om.readTree(frag);
          ok = fillFromNode(rl, node);
        } catch (Exception ignore) {
        }
      }
    }
    if (!ok) rl._llm_raw = Map.of("raw", content);
    return rl;
  }

  static boolean fillFromNode(ResultLine rl, JsonNode node) {
    if (node == null || !node.isObject()) return false;
    JsonNode ents = node.get("entities");
    JsonNode rels = node.get("relations");
    JsonNode intent = node.get("intent");
    if (ents != null && ents.isArray()) {
      for (JsonNode e : ents) {
        Map<String, Object> m = new LinkedHashMap<>();
        if (e.has("name")) m.put("name", e.get("name").asText(""));
        if (e.has("type")) m.put("type", e.get("type").asText("Entity"));
        if (!m.isEmpty()) rl.entities.add(m);
      }
    }
    if (rels != null && rels.isArray()) {
      for (JsonNode r : rels) {
        Map<String, Object> m = new LinkedHashMap<>();
        if (r.has("head")) m.put("head", r.get("head").asText(""));
        if (r.has("relation")) m.put("relation", r.get("relation").asText(""));
        if (r.has("tail")) m.put("tail", r.get("tail").asText(""));
        if (r.has("confidence")) m.put("confidence", r.get("confidence").asDouble(0.0));
        rl.relations.add(m);
      }
    }
    if (intent != null && intent.isTextual()) rl.intent = intent.asText("");
    return true;
  }

  /* ========================= Main ========================= */
  public static void main(String[] args) throws Exception {
    Args cfg = parseArgs(args);
    System.out.println("Input:  " + cfg.input);
    System.out.println("Output: " + cfg.output);
    List<Unit> units = buildUnits(cfg.input, cfg.maxUnits, cfg.keepQAPrefix, cfg.useNerHints, cfg.maxHints);
    System.out.println("Units:  " + units.size());
    if (units.isEmpty()) return;

    ObjectMapper om = new ObjectMapper();
    ExecutorService pool = Executors.newFixedThreadPool(cfg.threads);
    OllamaClient client = new OllamaClient(cfg.baseUrl, cfg.model, cfg.temperature);

    List<Future<List<ResultLine>>> futures = new ArrayList<>();
    for (int i = 0; i < units.size(); i += cfg.batchSize) {
      final int start = i;
      final int end = Math.min(units.size(), i + cfg.batchSize);
      List<Unit> batch = units.subList(start, end);
      futures.add(pool.submit(() -> callBatch(batch, client, cfg.maxRetries, om)));
    }

    try (OutputStream os = Files.newOutputStream(cfg.output); OutputStreamWriter w = new OutputStreamWriter(os,
        StandardCharsets.UTF_8); BufferedWriter bw = new BufferedWriter(w)) {
      for (Future<List<ResultLine>> f : futures) {
        List<ResultLine> lines = f.get();
        for (ResultLine rl : lines) {
          Map<String, Object> out = new LinkedHashMap<>();
          out.put("doc_id", rl.doc_id);
          out.put("chunk_id", rl.chunk_id);
          if (rl.question != null) out.put("question", rl.question);
          if (rl.answer != null) out.put("answer", rl.answer);
          if (rl.type != null) out.put("type", rl.type);
          out.put("text", rl.text);
          out.put("entities", rl.entities);
          out.put("relations", rl.relations);
          out.put("intent", rl.intent);
          if (rl._llm_raw != null) out.put("_llm_raw", rl._llm_raw);
          bw.write(om.writeValueAsString(out));
          bw.write(" ");
        }
        bw.flush();
      }
    }

    pool.shutdown();
    client.close();
    System.out.println("DONE → " + cfg.output);
  }

  static List<ResultLine> callBatch(List<Unit> batch, OllamaClient client, int maxRetries, ObjectMapper om) {
    List<ResultLine> out = new ArrayList<>();
    for (Unit u : batch) {
      String sys = SYSTEM_PROMPT;
      String user = buildUserPrompt(u);
      int tries = 0;
      String content = null;
      Exception last = null;
      while (tries++ < Math.max(1, maxRetries)) {
        try {
          Map<String, Object> resp = client.chatJSON(sys, user);
          Object msg = resp.get("message");
          if (msg instanceof Map) {
            Object c = ((Map<?, ?>) msg).get("content");
            if (c != null) content = String.valueOf(c);
          }
          if (content != null && !content.isBlank()) break;
        } catch (Exception e) {
          last = e;
        }
        try {
          Thread.sleep(500L * tries);
        } catch (InterruptedException ignored) {
        }
      }
      if (content == null) content = "{}";
      ResultLine rl = normalizeLLM(u, content, om);
      // Apply normalization / fixing rules
      postProcess(rl);
      out.add(rl);
    }
    return out;
  }
}
