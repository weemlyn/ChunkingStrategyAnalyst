package com.xwm.chunkingstrategyanalyst.entity;

import java.util.List;

/**
 * QA测试对数据类
 * 用于存储问题、相关文档和参考答案的对应关系
 */
public class QAPair {
    private final String question;           // 测试问题
    private final List<String> relevantDocs; // 相关文档ID列表
    private final String referenceAnswer;    // 参考答案（用于评估生成质量）

    /**
     * 构造函数
     * @param question 测试问题
     * @param relevantDocs 相关文档ID列表
     * @param referenceAnswer 参考答案
     */
    public QAPair(String question, List<String> relevantDocs, String referenceAnswer) {
        this.question = question;
        this.relevantDocs = relevantDocs;
        this.referenceAnswer = referenceAnswer;
    }

    public String getQuestion() { return question; }
    public List<String> getRelevantDocs() { return relevantDocs; }
    public String getReferenceAnswer() { return referenceAnswer; }
}
