package ai.nlp;

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

/**
 * Step 1 (English prompt version):
 * Java + Ollama to extract (entities, relations, intent) from nlp_output.jsonl.
 * <p>
 * ✔ Compatible with your schema: {rowIndex, question, answer, type, sentences[].text, sentences[].tokens[] (word,
 * pos,lemma,ner), ...}
 * ✔ Builds a combined TEXT from Q/A (or sentences), keeps Q/A fields in output for traceability.
 * ✔ Adds optional NER HINTS from tokens[].ner != 'O'.
 * ✔ Calls Ollama /api/chat with an English, JSON-strict prompt.
 * ✔ Robust JSON parsing (code-fence/trailing comma fixes), multi-thread + batching + retries.
 * <p>
 * Build:
 * mvn -q -DskipTests package
 * Run (example):
 * java -jar target/ollama-extractor-1.1.jar \
 * --input /mnt/data/nlp_output.jsonl \
 * --output /mnt/data/llm_structured_1.jsonl \
 * --model qwen2.5:7b-instruct-q4_K_M \
 * --baseUrl <a href="http://localhost:11434">...</a> \
 * --threads 6 --batchSize 8 --maxRetries 3 --temperature 0.0 \
 * --keepQAPrefix true --useNerHints true --maxHints 30
 */
public class OllamaExtractor {

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
    Path input = Path.of(a.getOrDefault("input", path + "nlp_output_2.jsonl"));
    Path output = Path.of(a.getOrDefault("output", path + "llm_structured_2.jsonl"));
    String model = a.getOrDefault("model", "qwen2.5:7b-instruct-q4_K_M");
    String base = a.getOrDefault("baseUrl", "http://localhost:11434");
    int threads = Integer.parseInt(a.getOrDefault("threads", "8"));
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

  /* Build Units from your JSONL schema exactly. */
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
          combined = (qPart + '\n' + aPart).trim();
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

  /* ========================= Ollama client ========================= */
  static final MediaType JSON = MediaType.parse("application/json; charset=utf-8");

  static class OllamaClient implements Closeable {
    final OkHttpClient http;
    final String baseUrl;
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
    public void close() {
    }
  }

  /* ========================= Prompt (EN) ========================= */
  // --- Controlled vocabularies ---
  static final List<String> ONTOLOGY = Arrays.asList(
      "APP", "PLAN", "BENEFIT", "PAYMENT_METHOD", "FEATURE", "ORG", "PRODUCT"
  );

  static final List<String> RELATIONS = Arrays.asList(
      "supports", "enables", "includes", "accepts_method", "belongs_to", "uses"
  );
  static final String SYSTEM_PROMPT = (
      "You are a precise information extraction system for building a knowledge graph. "
          + "From the given TEXT (may include Question/Answer) and optional HINTS (from NER), "
          + "produce ONLY a valid JSON object following the SCHEMA. No explanations, no prose, no code fences.\n\n"
          + "Requirements:\n"
          + "1) Extract all relevant entities (products, apps, services, features, payment methods, plans, benefits, " +
          "etc.). "
          + "Each entity must have {name, type}.\n"
          + "2) Detect and label generic FEATURES automatically. Whenever the text describes an action, service, or " +
          "capability "
          + "offered by the app, plan, or system (e.g., auto-pay, data sharing, call forwarding, roaming, blocking, " +
          "upgrade), "
          + "treat it as an entity of type FEATURE.\n"
          + "3) Infer feature names from actions if needed (e.g., 'enable roaming' → 'roaming'; 'set up auto-pay' → " +
          "'auto-pay').\n"
          + "4) Split coordinated mentions: if an entity name contains 'X or Y' or 'X and Y', split them into " +
          "separate entities.\n"
          + "5) Assign entity types from the allowed ontology: [APP, PLAN, FEATURE, BENEFIT, PAYMENT_METHOD, ORG, " +
          "PRODUCT]. "
          + "Map loosely if needed.\n"
          + "6) Build relations between entities with {head, relation, tail, confidence}. "
          + "Use relations from [supports, enables, includes, accepts_method, belongs_to, uses].\n"
          + "7) If a FEATURE is used by an APP, prefer relation 'supports'. "
          + "If a FEATURE involves a PAYMENT_METHOD, prefer 'accepts_method'. "
          + "If a PLAN lists included perks, use 'includes'.\n"
          + "8) Confidence must be a number in [0,1]. Default 0.75 if unsure.\n"
          + "9) Provide a concise, task-oriented 'intent' field (e.g., 'auto_payment_setup', 'plan_benefits_query').\n"
          + "10) Return JSON only, strictly valid according to the schema below.\n\n"
          + "Additional soft rules for better consistency:\n"
          + "11) Direction consistency: ensure relation direction follows natural semantics. "
          + "Provider → capability (APP → FEATURE), capability → requirement (FEATURE → PAYMENT_METHOD), "
          + "plan → included item (PLAN → BENEFIT/PRODUCT).\n"
          + "12) Type resolution: if a term ends with 'credits', 'allowance', 'bonus', 'benefit', or 'sharing', "
          + "prefer type BENEFIT instead of PRODUCT.\n"
          + "13) Default hierarchy: ORG may own APPs or PLANs (ORG → owns → APP/PLAN). "
          + "If text implies company ownership or offering, use that relation.\n"
          + "14) When uncertain between FEATURE and BENEFIT, use FEATURE if it’s an active action/configuration, "
          + "BENEFIT if it’s a passive advantage or inclusion.\n\n"
          + "Additional guidance:\n"
          + "- BENEFIT: perks or included advantages of a plan or service "
          + "(e.g., free data, bonus minutes, family sharing, streaming credits, allowances).\n"
          + "- FEATURE: capabilities or configurable functions (e.g., auto-pay, data sharing, call forwarding, " +
          "roaming, upgrade, blocking).\n"
          + "- PRODUCT: tangible or purchasable items (e.g., SIM card, router, handset). "
          + "When unclear, prefer BENEFIT/FEATURE if it’s not clearly physical.\n"
          + "15) Feature consistency: if the text describes an action or setup (e.g., 'set up', 'enable', 'activate'," +
          " 'manage'), "
          + "ensure the action is represented as a FEATURE even if implicit in the answer (e.g., 'Set up auto-pay' → " +
          "feature 'auto-pay').\n"
          + "16) Payment semantics: if the text discusses 'prepaid' or 'postpaid' options, treat them as PLAN or " +
          "PAYMENT_TYPE entities, "
          + "and use relation 'uses' or 'belongs_to' instead of 'accepts_method' when more semantically correct.\n"
  );

  static String buildUserPrompt(Unit u) {
    StringBuilder sb = new StringBuilder();

    sb.append("TEXT:\n").append(u.text).append("\n\n");

    if (u.hints != null && !u.hints.isEmpty()) {
      sb.append("HINTS (NER):\n");
      for (Hint h : u.hints) {
        sb.append("- ").append(h.name).append(" :: ").append(h.type).append("\n");
      }
      sb.append("\n");
    }

    if (u.type != null && !u.type.isBlank()) {
      sb.append("CATEGORY (from source): ").append(u.type).append("\n\n");
    }

    sb.append("Allowed entity.type (ontology): ").append(String.join(", ", ONTOLOGY)).append("\n");
    sb.append("Allowed relation set: ").append(String.join(", ", RELATIONS)).append("\n\n");

    sb.append("Instructions:\n");
    sb.append("- Extract entities with {name, type}. Use the ontology above; map loosely if needed (choose the " +
        "closest type).\n");
    sb.append("- If a mention contains coordination (e.g., 'bank account or credit card'), split into two entities "
        + "('bank account', 'credit card'). Do NOT include 'or/and' in names.\n");
    sb.append("- Build relations with {head, relation, tail, confidence}. Use only the allowed relations; "
        + "head/tail must be exact matches of entity names.\n");
    sb.append("- Set confidence in [0,1]; if plausible but not fully certain, use 0.75.\n");
    sb.append("- Provide a concise intent label (snake_case), e.g., 'auto_payment_setup', 'plan_benefits_query'.\n");
    sb.append("- Return JSON only, no extra text.\n\n");

    sb.append("SCHEMA (return exactly this JSON shape):\n");
    sb.append("{\n");
    sb.append("  \"entities\": [{\"name\": \"...\", \"type\": \"...\"}],\n");
    sb.append("  \"relations\": [{\"head\": \"...\", \"relation\": \"...\", \"tail\": \"...\", \"confidence\": 0.0}]," +
        "\n");
    sb.append("  \"intent\": \"...\"\n");
    sb.append("}\n");

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

  static ResultLine normalizeLLM(Unit u, String content, ObjectMapper om) {
    ResultLine rl = new ResultLine();
    rl.doc_id = u.docId;
    rl.chunk_id = u.chunkId;
    rl.question = u.question;
    rl.answer = u.answer;
    rl.type = u.type;
    rl.text = u.text;
    String t = content.trim();
    t = t.replaceAll("^```json *", "");
    t = t.replaceAll("^``` *", "");
    t = t.replaceAll("```$", "");
    t = t.replace('\'', '"');
    t = t.replaceAll(", *([}\\]])", "$1");
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
      out.add(rl);
    }
    return out;
  }

  public static void main(String[] args) throws Exception {
    Args cfg = parseArgs(args);
    System.out.println("Input: " + cfg.input);
    System.out.println("Output: " + cfg.output);
    List<Unit> units = buildUnits(cfg.input, cfg.maxUnits, cfg.keepQAPrefix, cfg.useNerHints, cfg.maxHints);
    System.out.println("Units: " + units.size());
    if (units.isEmpty()) return;

    ObjectMapper om = new ObjectMapper();
    ExecutorService pool = Executors.newFixedThreadPool(cfg.threads);
    OllamaClient client = new OllamaClient(cfg.baseUrl, cfg.model, cfg.temperature);

    List<Future<List<ResultLine>>> futures = new ArrayList<>();
    for (int start = 0; start < units.size(); start += cfg.batchSize) {
      final int end = Math.min(units.size(), start + cfg.batchSize);
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
          bw.write("\n");
        }
        bw.flush();
      }
    }
    pool.shutdown();
    client.close();
    System.out.println("DONE → " + cfg.output);
  }

}