package ai.nlp;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations;
import edu.stanford.nlp.semgraph.SemanticGraphEdge;
import edu.stanford.nlp.util.CoreMap;
import org.apache.poi.ss.usermodel.*;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;

import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.regex.Pattern;

public class NLPExtractor {

  static class TokenInfo {
    String word, pos, lemma, ner;

    public TokenInfo(String word, String pos, String lemma, String ner) {
      this.word = word;
      this.pos = pos;
      this.lemma = lemma;
      this.ner = ner;
    }
  }

  static class DepInfo {
    String relation, governor, dependent;

    public DepInfo(String relation, String governor, String dependent) {
      this.relation = relation;
      this.governor = governor;
      this.dependent = dependent;
    }
  }

  static class SentenceInfo {
    String text;
    List<TokenInfo> tokens = new ArrayList<>();
    List<DepInfo> dependencies = new ArrayList<>();
  }

  static class RowNLPRecord {
    int rowIndex;
    String question;
    String answer;
    String type;
    List<SentenceInfo> sentences = new ArrayList<>();
  }

  public static void main(String[] args) throws Exception {
    // Input/Output
    String inputXlsx = args.length > 0 ? args[0] :
        "D:\\project\\embedding-service\\src\\main\\java\\ai\\nlp\\input" + "\\uplus_10000.xlsx";
    String outputJsonl = args.length > 1 ? args[1] : "D:\\project\\embedding-service\\src\\main\\java\\ai\\nlp" +
        "\\output\\nlp_output.jsonl";

    // 1) Core NLP pipeline setup
    Properties props = new Properties();
    props.setProperty("annotators", "tokenize,ssplit,pos,lemma,ner,regexner,parse");
    props.setProperty("tokenize.language", "en");
    props.setProperty("regexner.mapping", "D:\\project\\embedding-service\\src\\main\\java\\ai\\nlp" +
        "\\regexner_gazetteer.tsv");
    props.setProperty("regexner.ignorecase", "true");
    props.setProperty("regexner.noDefaultOverwriteLabels", "O");

    // speed hints (optional)
    props.setProperty("threads", String.valueOf(Math.max(1, Runtime.getRuntime().availableProcessors() - 1)));
    StanfordCoreNLP pipeline = new StanfordCoreNLP(props);

    // 2) Read Excel file
    try (FileInputStream fis = new FileInputStream(inputXlsx); Workbook workbook = new XSSFWorkbook(fis); BufferedWriter out = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outputJsonl), StandardCharsets.UTF_8))) {

      Sheet sheet = workbook.getSheetAt(0);

      // Map header => column index
      Map<String, Integer> col = mapHeader(sheet.getRow(0));
      mustHave(col, "question");
      mustHave(col, "answer");

      Gson gson = new GsonBuilder().disableHtmlEscaping().create();
      int rowCount = 0;
      for (Row row : sheet) {
        if (row.getRowNum() == 0) continue;
        String question = getCellString(row.getCell(col.get("question")));
        String answer = getCellString(row.getCell(col.get("answer")));
        String type = col.containsKey("type") ? getCellString(row.getCell(col.get("type"))) : "";

        if ((question == null || question.isBlank()) && (answer == null || answer.isBlank())) continue;
        String rawText = "Q: " + (question == null ? "" : question) + " A: " + (answer == null ? "" : answer);
        String text = preprocessText(rawText);

        // 3) Annotate
        Annotation doc = new Annotation(text);
        pipeline.annotate(doc);

        RowNLPRecord record = new RowNLPRecord();
        record.rowIndex = row.getRowNum();
        record.question = question;
        record.answer = answer;
        record.type = type;

        for (CoreMap sentence : doc.get(CoreAnnotations.SentencesAnnotation.class)) {
          SentenceInfo sInfo = new SentenceInfo();
          String sentText = sentence.get(CoreAnnotations.TextAnnotation.class);
          sInfo.text = postprocessToken(sentText);

          for (CoreLabel token : sentence.get(CoreAnnotations.TokensAnnotation.class)) {
            String w = postprocessToken(token.word());
            String lemma = postprocessToken(token.lemma());
            sInfo.tokens.add(new TokenInfo(w, token.get(CoreAnnotations.PartOfSpeechAnnotation.class), lemma,
                token.ner()));
          }

          SemanticGraph deps = sentence.get(SemanticGraphCoreAnnotations.EnhancedPlusPlusDependenciesAnnotation.class);
          for (SemanticGraphEdge e : deps.edgeListSorted()) {
            String gov = postprocessToken(e.getGovernor().word());
            String dep = postprocessToken(e.getDependent().word());
            sInfo.dependencies.add(new DepInfo(
                e.getRelation().toString(),
                gov,
                dep
            ));
          }
          record.sentences.add(sInfo);
        }

        out.write(gson.toJson(record));
        out.newLine();

        rowCount++;
        if (rowCount % 200 == 0) {
          System.out.println("Processed rows: " + rowCount);
        }
      }
    }
    System.out.println("✅ Done. JSONL saved to: " + outputJsonl);
  }

  // --- Helper methods (mapHeader, processText, extractSentenceInfo, toJson) would go here ---
  private static Map<String, Integer> mapHeader(Row headerRow) {
    Map<String, Integer> map = new HashMap<>();
    if (headerRow == null) return map;
    for (Cell cell : headerRow) {
      String name = getCellString(cell);
      if (name != null) map.put(name.trim().toLowerCase(), cell.getColumnIndex());
    }
    return map;
  }

  private static String getCellString(Cell cell) {
    if (cell == null) return "";
    return switch (cell.getCellType()) {
      case STRING -> cell.getStringCellValue();
      case NUMERIC -> DateUtil.isCellDateFormatted(cell) ? cell.getDateCellValue().toString() :
          (cell.getNumericCellValue() == Math.floor(cell.getNumericCellValue()) ?
              String.valueOf((long) cell.getNumericCellValue()) : String.valueOf(cell.getNumericCellValue()));
      case BOOLEAN -> String.valueOf(cell.getBooleanCellValue());
      case FORMULA -> {
        try {
          yield cell.getStringCellValue();
        } catch (Exception e) {
          yield String.valueOf(cell.getNumericCellValue());
        }
      }
      default -> null;
    };
  }

  private static void mustHave(Map<String, Integer> col, String key) {
    if (!col.containsKey(key)) {
      throw new IllegalArgumentException("Missing required column: " + key);
    }
  }

  private static String postprocessToken(String w) {
    return w.replace("U_PLUS", "U+");
  }

  private static final Pattern RX_U_PLUS = Pattern.compile("(?i)(?<!\\w)u\\s*\\+");

  private static String preprocessText(String s) {
    if (s == null || s.isEmpty()) return s;
    return RX_U_PLUS.matcher(s).replaceAll("U_PLUS");
  }

}
