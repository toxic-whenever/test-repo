package ai.nlp.service;// IngestQA.java

import org.apache.poi.ss.usermodel.*;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;
import org.neo4j.driver.*;

import java.io.FileInputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.neo4j.driver.Values.parameters;

public class IngestQA {
  public static void main(String[] args) throws Exception {
    String excel = args.length > 0 ? args[0] : "D:\\project\\embedding-service\\src\\main\\java\\ai\\nlp\\input\\uplus_10000_dedup.xlsx";
    String teiUrl = System.getenv().getOrDefault("TEI_URL", "http://localhost:8080/embeddings");

    String uri = System.getenv().getOrDefault("NEO4J_URI", "bolt://localhost:7687");
    String user = System.getenv().getOrDefault("NEO4J_USER", "neo4j");
    String pass = System.getenv().getOrDefault("NEO4J_PASS", "12345678");
    String db = System.getenv().getOrDefault("NEO4J_DB", "rag");

    try (Driver driver = GraphDatabase.driver(uri, AuthTokens.basic(user, pass));
         Session session = driver.session(SessionConfig.forDatabase(db));
         FileInputStream fis = new FileInputStream(excel);
         Workbook wb = new XSSFWorkbook(fis)) {

      Sheet sh = wb.getSheetAt(0);
      Row header = sh.getRow(0);
      if (header == null) throw new IllegalStateException("Thiếu header");

      Map<String, Integer> col = detect(header);
      int qCol = col.get("question"), aCol = col.get("answer");

      EmbeddingClient embedder = new EmbeddingClient(teiUrl);
      DataFormatter fmt = new DataFormatter();

      final int BATCH = 500;
      List<Map<String, Object>> batch = new ArrayList<>(BATCH);

      for (int r = 1; r <= sh.getLastRowNum(); r++) {
        Row row = sh.getRow(r);
        if (row == null) continue;
        String q = fmt.formatCellValue(row.getCell(qCol)).trim();
        String a = fmt.formatCellValue(row.getCell(aCol)).trim();
        if (q.isBlank() || a.isBlank()) continue;

        // embed nội dung (passage)
        float[] emb = embedder.embedE5(q + " " + a, false); // false = passage
        List<Double> vec = new ArrayList<>(emb.length);
        for (float f : emb) vec.add((double) f);

        Map<String, Object> m = new HashMap<>();
        m.put("id", "qa-" + r); // hoặc lấy từ cột id nếu có
        m.put("q", q);
        m.put("a", a);
        m.put("emb", vec);
        batch.add(m);

        if (batch.size() >= BATCH) {
          write(session, batch);
          batch.clear();
        }
      }
      if (!batch.isEmpty()) write(session, batch);

      System.out.println("✅ Ingest xong");
    }
  }

  static void write(Session s, List<Map<String, Object>> rows) {
    s.executeWrite(tx -> {
      tx.run("""
            UNWIND $rows AS row
            MERGE (n:QA {id: row.id})
            SET n.question=row.q, n.answer=row.a, n.embedding=row.emb
          """, parameters("rows", rows));
      return null;
    });
  }

  static Map<String, Integer> detect(Row header) {
    Map<String, Integer> m = new HashMap<>();
    for (Cell c : header) {
      String name = c == null ? "" : c.getStringCellValue();
      String low = name == null ? "" : name.trim().toLowerCase();
      if (low.contains("question")) m.put("question", c.getColumnIndex());
      if (low.contains("answer")) m.put("answer", c.getColumnIndex());
    }
    if (!m.containsKey("question") || !m.containsKey("answer"))
      throw new IllegalStateException("Không tìm thấy cột question/answer");
    return m;
  }
}