package ai.nlp.service;

import org.neo4j.driver.*;

import java.util.*;

import static org.neo4j.driver.Values.parameters;


public class QueryANN {

  public static void main(String[] args) throws Exception {
    String userQuery = String.join(" ", args).trim();
    if (userQuery.isBlank()) userQuery = "What does the 5G Basic include?";

    EmbeddingClient embedder = new EmbeddingClient("http://localhost:8080/embeddings");
    float[] qemb = embedder.embedE5(userQuery, true);
    List<Double> vec = new ArrayList<>(qemb.length);
    for (float f : qemb) vec.add((double) f);

    try (Driver d = GraphDatabase.driver("bolt://localhost:7687",
        AuthTokens.basic("neo4j", "12345678"));
         Session s = d.session(SessionConfig.forDatabase("rag"))) {

      var result = s.run("""
            WITH $qemb AS qemb
            CALL db.index.vector.queryNodes('qa_embedding_index', 10, qemb)
            YIELD node, score
            RETURN node.id AS id, node.question AS question, node.answer AS answer, score
            ORDER BY score DESC LIMIT 5
          """, parameters("qemb", vec));

      while (result.hasNext()) {
        var rec = result.next();
        System.out.printf("• %s | score=%.4f\nQ: %s\nA: %s\n\n",
            rec.get("id").asString(),
            rec.get("score").asDouble(),
            rec.get("question").asString(),
            rec.get("answer").asString());
      }
    }
  }

}
