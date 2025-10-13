package ai.nlp.service;

public interface LlmClient {
  String complete(String prompt) throws Exception;
}

