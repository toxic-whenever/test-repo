package ai.nlp;

import org.apache.poi.ss.usermodel.*;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;

import java.io.*;
import java.util.*;

/**
 * Khử trùng các dòng trong Excel dựa trên (question, answer),
 * giữ lại lần xuất hiện đầu tiên, ghi ra file mới mà vẫn giữ nguyên toàn bộ cột.
 * <p>
 * Cách xác định cột:
 * - Tự động dò theo tên header: cột chứa "question" và "answer" (không phân biệt hoa thường, chấp nhận chứa từ).
 * - Có thể chỉ định tên cột qua args nếu muốn (xem phần main).
 * <p>
 * Mặc định xử lý sheet đầu tiên.
 */
public class DedupQuestionAnswerPreserveStructure {

  public static void main(String[] args) {
    // Tham số CLI (tuỳ chọn):
    // args[0] = inputPath, args[1] = outputPath, args[2] = questionHeaderName, args[3] = answerHeaderName
    String inputPath = args.length > 0 ? args[0] : "D:\\project\\embedding-service\\src\\main\\java\\ai\\nlp\\input" +
        "\\" + "uplus_10000.xlsx";
    String outputPath = args.length > 1 ? args[1] : "D:\\project\\embedding-service\\src\\main\\java\\ai\\nlp\\input" +
        "\\" + "uplus_10000_dedup.xlsx";
    String qHeaderNameOverride = args.length > 2 ? args[2] : null;
    String aHeaderNameOverride = args.length > 3 ? args[3] : null;

    try (FileInputStream fis = new FileInputStream(inputPath);
         Workbook inWb = new XSSFWorkbook(fis)) {

      Sheet inSheet = inWb.getSheetAt(0);
      if (inSheet == null) {
        throw new IllegalStateException("Không tìm thấy sheet đầu tiên.");
      }

      Row header = inSheet.getRow(0);
      if (header == null) {
        throw new IllegalStateException("Không tìm thấy hàng header (row 0).");
      }

      // Số cột cần giữ nguyên cấu trúc
      int colCount = header.getLastCellNum();

      // Tìm cột question/answer
      int qCol = findColumnIndex(header, qHeaderNameOverride, List.of("question", "ques", "query", "prompt", "cauhoi"
          , "hoi"));
      int aCol = findColumnIndex(header, aHeaderNameOverride, List.of("answer", "ans", "response", "completion",
          "traloi", "tra_loi", "reply"));

      if (qCol < 0 || aCol < 0) {
        throw new IllegalStateException("Không tìm thấy cột 'question' hoặc 'answer' trong header.");
      }

      DataFormatter formatter = new DataFormatter();
      FormulaEvaluator evaluator = inWb.getCreationHelper().createFormulaEvaluator();

      // Dùng LinkedHashSet để vừa tra cứu nhanh, vừa giữ thứ tự lần đầu gặp
      Set<String> seen = new LinkedHashSet<>();

      // Workbook output
      Workbook outWb = new XSSFWorkbook();
      Sheet outSheet = outWb.createSheet(inSheet.getSheetName() + "_dedup");

      // Ghi header
      Row outHeader = outSheet.createRow(0);
      for (int c = 0; c < colCount; c++) {
        copyCellValue(header.getCell(c), outHeader.createCell(c), formatter, evaluator);
      }

      int lastRow = inSheet.getLastRowNum();
      int outRowIdx = 1;
      int removedDup = 0;
      int removedEmpty = 0;
      int kept = 0;

      for (int r = 1; r <= lastRow; r++) {
        Row row = inSheet.getRow(r);
        if (row == null) continue;

        String qRaw = getCellAsDisplayString(row.getCell(qCol), formatter, evaluator);
        String aRaw = getCellAsDisplayString(row.getCell(aCol), formatter, evaluator);

        String qKey = normalizeForKey(qRaw);
        String aKey = normalizeForKey(aRaw);

        boolean eligible = !qKey.isEmpty() && !aKey.isEmpty();

        if (!eligible) {
          // Không đủ điều kiện gom trùng ⇒ giữ nguyên
          Row outRow = outSheet.createRow(outRowIdx++);
          for (int c = 0; c < colCount; c++) {
            copyCellValue(row.getCell(c), outRow.createCell(c), formatter, evaluator);
          }
          kept++;
          continue;
        }

        String key = qKey + "||" + aKey;

        if (seen.contains(key)) {
          // Đã có cặp (q,a) trước đó ⇒ xoá lần sau
          removedDup++;
        } else {
          // Lần đầu tiên ⇒ giữ lại và đánh dấu đã thấy
          seen.add(key);
          Row outRow = outSheet.createRow(outRowIdx++);
          for (int c = 0; c < colCount; c++) {
            copyCellValue(row.getCell(c), outRow.createCell(c), formatter, evaluator);
          }
          kept++;
        }
      }

      // Tuỳ chọn: autosize các cột (có thể tốn thời gian với file lớn)
      for (int c = 0; c < colCount; c++) {
        try {
          outSheet.autoSizeColumn(c);
        } catch (Exception ignore) {
        }
      }

      try (FileOutputStream fos = new FileOutputStream(outputPath)) {
        outWb.write(fos);
      }
      outWb.close();

      int originalRows = lastRow + 1;
      System.out.println("✅ Hoàn tất khử trùng theo (question, answer).");
      System.out.println("Tổng số dòng gốc (kể cả header): " + (originalRows));
      System.out.println("Giữ lại: " + kept + " dòng (không rỗng hoặc không trùng).");
      System.out.println("Xoá do trùng: " + removedDup + " dòng.");
      System.out.println("📁 File kết quả: " + outputPath);

    } catch (Exception e) {
      e.printStackTrace();
    }
  }

  /**
   * Tìm index cột. Ưu tiên tên override; nếu không có thì dò theo danh sách alias.
   */
  private static int findColumnIndex(Row header, String overrideName, List<String> aliases) {
    int colCount = header.getLastCellNum();
    // 1) Nếu có overrideName: match không phân biệt hoa thường, match “chứa”
    if (overrideName != null && !overrideName.isBlank()) {
      String needle = overrideName.trim().toLowerCase();
      for (int c = 0; c < colCount; c++) {
        String name = safeLower(header.getCell(c));
        if (name.contains(needle)) return c;
      }
    }
    // 2) Dò theo alias
    for (int c = 0; c < colCount; c++) {
      String name = safeLower(header.getCell(c));
      if (name == null) continue;
      for (String a : aliases) {
        if (name.contains(a)) return c;
      }
    }
    return -1;
  }

  private static String safeLower(Cell cell) {
    if (cell == null) return null;
    String s = cell.getCellType() == CellType.STRING ? cell.getStringCellValue() : cell.toString();
    return s == null ? null : s.trim().toLowerCase();
  }

  /**
   * Lấy chuỗi hiển thị giống Excel (DataFormatter) để hạn chế lỗi format số/ngày.
   */
  private static String getCellAsDisplayString(Cell cell, DataFormatter formatter, FormulaEvaluator evaluator) {
    if (cell == null) return "";
    return formatter.formatCellValue(cell, evaluator);
  }

  /**
   * Chuẩn hoá tạo key: trim, gộp khoảng trắng, lowercase.
   */
  private static String normalizeForKey(String s) {
    if (s == null) return "";
    String t = s.trim().replaceAll("\\s+", " ");
    return t.toLowerCase();
  }

  /**
   * Copy giá trị từ cell nguồn sang cell đích (value only, không copy style để đơn giản/stable).
   */
  private static void copyCellValue(Cell src, Cell tgt, DataFormatter formatter, FormulaEvaluator evaluator) {
    if (src == null) {
      tgt.setBlank();
      return;
    }
    CellType type = src.getCellType();
    if (type == CellType.FORMULA) {
      // Với công thức: ghi ra giá trị hiển thị tương tự Excel
      String display = formatter.formatCellValue(src, evaluator);
      tgt.setCellValue(display);
      return;
    }
    switch (type) {
      case STRING:
        tgt.setCellValue(src.getStringCellValue());
        break;
      case NUMERIC:
        if (DateUtil.isCellDateFormatted(src)) {
          // Ghi dạng số serial hoặc dạng chuỗi định dạng? Để dễ đọc: ghi đúng hiển thị của Excel
          String display = formatter.formatCellValue(src);
          tgt.setCellValue(display);
        } else {
          // Ghi số thực tế (tránh khoa học). Nếu muốn giữ y nguyên hiển thị => dùng formatter
          String display = formatter.formatCellValue(src);
          // Thử parse về số nếu có thể, nếu không thì ghi chuỗi hiển thị
          try {
            double d = Double.parseDouble(display.replace(",", ""));
            tgt.setCellValue(d);
          } catch (NumberFormatException e) {
            tgt.setCellValue(display);
          }
        }
        break;
      case BOOLEAN:
        tgt.setCellValue(src.getBooleanCellValue());
        break;
      case BLANK:
        tgt.setBlank();
        break;
      default:
        // Các loại khác: ghi theo hiển thị
        String display = formatter.formatCellValue(src);
        tgt.setCellValue(display);
    }
  }
}