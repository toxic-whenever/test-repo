package ai.nlp.service;

public class DummyLlm implements LlmClient {
  public String complete(String prompt) {
    return "Đây là câu trả lời dựa trên ngữ cảnh đã chọn.\n\n" + prompt;
  }
}