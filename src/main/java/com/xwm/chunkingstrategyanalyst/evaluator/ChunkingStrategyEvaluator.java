package com.xwm.chunkingstrategyanalyst.evaluator;
import com.xwm.chunkingstrategyanalyst.entity.EvaluationMetrics;
import com.xwm.chunkingstrategyanalyst.entity.QAPair;
import dev.langchain4j.community.model.dashscope.QwenChatModel;
import dev.langchain4j.data.document.*;
import dev.langchain4j.data.document.splitter.*;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;

import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingSearchResult;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import java.util.*;
import java.util.stream.Collectors;

public class ChunkingStrategyEvaluator {
    private final List<Document> documents;  // 待处理的文档列表
    private final List<QAPair> qaPairs;      // 测试用的QA对列表
    private final EmbeddingModel embeddingModel;  // 嵌入模型，用于生成向量
    private final QwenChatModel chatModel;      // 聊天模型，用于生成回答
    private final Map<String, String> segmentTextToDocIdMap;  // 文本片段内容到文档ID的映射

    /**
     * 构造函数
     * @param documents 待评估的文档列表
     * @param qaPairs 用于评估的QA测试对
     */
    public ChunkingStrategyEvaluator(List<Document> documents, List<QAPair> qaPairs) {
        this.documents = documents;
        this.qaPairs = qaPairs;
        this.segmentTextToDocIdMap = new HashMap<>();  // 初始化片段文本到文档ID的映射

        // 初始化嵌入模型 - 使用轻量级的AllMiniLM模型
        this.embeddingModel = new AllMiniLmL6V2EmbeddingModel();


        // 初始化聊天模型 - 用于生成回答和评估
        this.chatModel = QwenChatModel.builder()
                .modelName("qwen3-max")
                .apiKey("sk-196000a734cd4450a28ec874781bcdec")
                .build();
    }

    /**
     * 运行全面评估
     * 对定义的所有chunking策略进行完整的评估流程
     * @return 包含所有策略评估结果的列表
     */
    public List<EvaluationMetrics> runComprehensiveEvaluation() {
        // 获取所有定义的策略
        Map<String, DocumentSplitter> strategies = this.defineStrategies();
        List<EvaluationMetrics> allMetrics = new ArrayList<>();

        System.out.println("开始全面评估，共 " + strategies.size() + " 种策略");

        // 对每种策略进行评估
        for (Map.Entry<String, DocumentSplitter> entry : strategies.entrySet()) {
            String strategyName = entry.getKey();
            DocumentSplitter splitter = entry.getValue();

            System.out.println("\n" + "=".repeat(60));
            System.out.println("正在评估策略: " + strategyName);
            System.out.println("=".repeat(60));

            // 步骤1: 应用chunking策略分割文档
            long buildStart = System.currentTimeMillis();
            List<TextSegment> segments = this.applyChunkingStrategy(strategyName, splitter);
            double buildTime = (System.currentTimeMillis() - buildStart) / 1000.0;

            // 步骤2: 构建向量存储
            EmbeddingStore<TextSegment> vectorStore = this.buildVectorStore(segments, strategyName);

            // 步骤3: 评估检索性能
            Map<String, Object> retrievalMetrics = this.evaluateRetrieval(vectorStore, strategyName);

            // 步骤4: 评估生成性能
            Map<String, Double> generationMetrics = this.evaluateGeneration(vectorStore, strategyName);

            // 提取检索指标
            @SuppressWarnings("unchecked")
            Map<Integer, Double> recallAtK = (Map<Integer, Double>) retrievalMetrics.get("recallAtK");
            @SuppressWarnings("unchecked")
            Map<Integer, Double> precisionAtK = (Map<Integer, Double>) retrievalMetrics.get("precisionAtK");

            // 创建评估指标对象（NDCG暂时设为0，需要更复杂的实现）
            EvaluationMetrics metrics = new EvaluationMetrics(
                    strategyName,
                    recallAtK,
                    precisionAtK,
                    (Double) retrievalMetrics.get("mrr"),
                    0.0, // NDCG 需要更复杂的排序质量评估实现
                    (Double) retrievalMetrics.get("latency"),
                    segments.size(),
                    generationMetrics.get("faithfulness"),
                    generationMetrics.get("contextUtilization"),
                    generationMetrics.get("hallucinationRate"),
                    buildTime
            );

            allMetrics.add(metrics);
            System.out.println("策略 " + strategyName + " 评估完成");
        }

        System.out.println("\n所有策略评估完成！");
        return allMetrics;
    }

    /**
     * 定义不同的chunking策略
     * 这里实现了文档中提到的几种主要分割策略
     * @return 策略名称到分割器的映射
     */
    public Map<String, DocumentSplitter> defineStrategies() {
        Map<String, DocumentSplitter> strategies = new HashMap<>();

        // 策略1: 固定大小分割 - 200 tokens，重叠20 tokens
        strategies.put("fixed_200_20", new DocumentByParagraphSplitter(200, 20));

        // 策略2: 固定大小分割 - 500 tokens，重叠50 tokens
        strategies.put("fixed_500_50", new DocumentByParagraphSplitter(500, 50));

        // 策略3: 递归分割 - 500 tokens，重叠50 tokens
        strategies.put("recursive_500_50", DocumentSplitters.recursive(500, 50));

        // 策略4: 按句子分割 - 300 tokens，重叠30 tokens
        strategies.put("sentence_based_300_30", new DocumentBySentenceSplitter(300, 30));

        // 策略5: 滑动窗口 - 500 tokens，重叠100 tokens
        strategies.put("sliding_window_500_100", new DocumentByParagraphSplitter(500, 100));

        return strategies;
    }

    /**
     * 应用指定的chunking策略分割文档
     * @param strategyName 策略名称
     * @param splitter 文档分割器
     * @return 分割后的文本片段列表
     */
    public List<TextSegment> applyChunkingStrategy(String strategyName, DocumentSplitter splitter) {
        System.out.println("正在应用策略: " + strategyName);
        long startTime = System.currentTimeMillis();

        // 对每个文档应用分割策略
        List<TextSegment> segments = new ArrayList<>();
        segmentTextToDocIdMap.clear();  // 清空之前的映射，为新策略准备
        for (int i = 0; i < documents.size(); i++) {
            Document doc = documents.get(i);
            // 生成文档ID（使用索引+1，对应ChunkingEvaluationDemo中的doc1, doc2, doc3）
            String docId = "doc" + (i + 1);
            
            // 分割文档
            List<TextSegment> docSegments = splitter.split(doc);
            
            // 为每个片段关联文档ID，并添加到列表
            for (TextSegment segment : docSegments) {
                segments.add(segment);
                // 使用文本内容作为key，将片段文本和文档ID的映射关系存储起来
                segmentTextToDocIdMap.put(segment.text(), docId);
            }
        }

        long endTime = System.currentTimeMillis();
        double buildTime = (endTime - startTime) / 1000.0;

        System.out.printf("生成 %d 个文本片段, 耗时: %.2f秒%n", segments.size(), buildTime);
        return segments;
    }

    /**
     * 构建向量存储
     * 将分割后的文本片段转换为向量并存储到向量数据库中
     * @param segments 文本片段列表
     * @param strategyName 策略名称（用于标识）
     * @return 向量存储实例
     */
    public EmbeddingStore<TextSegment> buildVectorStore(List<TextSegment> segments, String strategyName) {
        System.out.println("为策略 " + strategyName + " 构建向量存储...");

        // 使用内存向量存储（生产环境可替换为Milvus、Pinecone等）
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();

        // 为每个文本片段生成嵌入向量并存储
        for (TextSegment segment : segments) {
            var embedding = embeddingModel.embed(segment.text()).content();
            embeddingStore.add(embedding, segment);
        }

        System.out.println("向量存储构建完成，包含 " + segments.size() + " 个向量");
        return embeddingStore;
    }

    /**
     * 评估检索性能
     * 计算召回率、精确率、MRR等检索指标
     * @param embeddingStore 向量存储
     * @param strategyName 策略名称
     * @return 包含各项检索指标的映射
     */
    public Map<String, Object> evaluateRetrieval(EmbeddingStore<TextSegment> embeddingStore, String strategyName) {
        System.out.println("正在评估检索性能: " + strategyName);

        // 初始化指标存储
        Map<Integer, Double> recallScores = new HashMap<>();
        Map<Integer, Double> precisionScores = new HashMap<>();
        List<Double> reciprocalRanks = new ArrayList<>();  // 用于计算MRR

        // 定义要评估的K值
//        K值影响分析：
//        小K值：强调排名精度（相关文档是否靠前）
//        大K值：强调覆盖广度（能否找到所有相关文档）
        int[] kValues = {1, 3, 5, 10};
        for (int k : kValues) {
            recallScores.put(k, 0.0);
            precisionScores.put(k, 0.0);
        }

        double totalLatency = 0.0;
        int totalQueries = qaPairs.size();

        // 对每个QA对进行检索测试
        for (QAPair qaPair : qaPairs) {
            Embedding query = embeddingModel.embed(qaPair.getQuestion()).content();
            List<String> groundTruthDocs = qaPair.getRelevantDocs();
            EmbeddingSearchRequest embeddingSearchRequest = EmbeddingSearchRequest.builder()
                    .queryEmbedding(query)
                    .maxResults(10)//匹配最相似的10条记录
                    .minScore(0.0)
                    .build();
            // 测量检索延迟
            long startTime = System.nanoTime();
            EmbeddingSearchResult<TextSegment> relevant = embeddingStore.search(embeddingSearchRequest);
            long retrievalTime = System.nanoTime() - startTime;
            totalLatency += retrievalTime / 1_000_000.0; // 转换为毫秒
            //获取匹配结果
            List<EmbeddingMatch<TextSegment>> matches = relevant.matches();
            // 获取检索结果的文档ID（从segmentTextToDocIdMap映射中获取）
            List<String> retrievedIds = matches.stream()
                    .map(embeddingMatch -> {
                        TextSegment segment = embeddingMatch.embedded();
                        // 使用文本内容从映射中获取文档ID
                        String docId = segmentTextToDocIdMap.get(segment.text());
                        // 如果映射中没有找到，返回空字符串（这种情况不应该发生）
                        if (docId == null || docId.isEmpty()) {
                            return "";
                        }
                        return docId;
                    })
                    .filter(id -> !id.isEmpty()) // 过滤掉空的ID
                    .collect(Collectors.toList());

            // 计算不同K值下的召回率和精确率
            for (int k : kValues) {
                List<String> topKIds = retrievedIds.subList(0, Math.min(k, retrievedIds.size()));
                
                // 对检索结果中的文档ID去重（一个文档可能被分割成多个片段）
                // 召回率应该基于文档级别，而不是片段级别
                Set<String> uniqueRetrievedDocs = new LinkedHashSet<>(topKIds);
                
                // 计算去重后命中的相关文档数（文档级别）
                long hits = uniqueRetrievedDocs.stream()
                        .filter(groundTruthDocs::contains)
                        .count();

                // 召回率 = 检索到的相关文档数（去重后） / 总相关文档数
                // Recall@K: 在前K个检索结果中，有多少比例的相关文档被找到
                double recall = groundTruthDocs.isEmpty() ? 0.0 : (double) hits / groundTruthDocs.size();
                
                // 精确率 = 检索到的相关文档数（去重后） / 检索出的唯一文档数
                // Precision@K: 在前K个检索结果中，检索到的文档有多少比例是相关的
                double precision = uniqueRetrievedDocs.isEmpty() ? 0.0 : (double) hits / uniqueRetrievedDocs.size();

                recallScores.put(k, recallScores.get(k) + recall);
                precisionScores.put(k, precisionScores.get(k) + precision);
            }

            // 计算MRR（平均倒数排名）
            // MRR应该基于文档级别，需要找到第一个相关文档的位置
            Set<String> seenDocs = new HashSet<>();  // 记录已经见过的文档ID
            for (int i = 0; i < retrievedIds.size(); i++) {
                String docId = retrievedIds.get(i);
                // 如果是新的文档（之前没见过）且是相关文档
                if (seenDocs.add(docId) && groundTruthDocs.contains(docId)) {
                    // 计算该文档的排名（需要计算这是第几个唯一的文档）
                    int rank = seenDocs.size();  // 当前唯一文档的数量即为排名
                    reciprocalRanks.add(1.0 / rank);  // 第一个相关文档的排名倒数
                    break;
                }
            }
        }

        // 计算各项指标的平均值
        Map<Integer, Double> recallAtK = new HashMap<>();
        Map<Integer, Double> precisionAtK = new HashMap<>();
        for (int k : kValues) {
            recallAtK.put(k, recallScores.get(k) / totalQueries);
            precisionAtK.put(k, precisionScores.get(k) / totalQueries);
        }

        // 计算MRR和平均延迟
        double mrr = reciprocalRanks.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        double avgLatency = totalLatency / totalQueries;

        // 组装结果
        Map<String, Object> results = new HashMap<>();
        results.put("recallAtK", recallAtK);
        results.put("precisionAtK", precisionAtK);
        results.put("mrr", mrr);
        results.put("latency", avgLatency);

        System.out.printf("检索评估完成: Recall@5=%.3f, MRR=%.3f, 平均延迟=%.2fms%n",
                recallAtK.get(5), mrr, avgLatency);

        return results;
    }

    /**
     * 评估生成性能
     * 评估LLM基于检索内容生成回答的质量
     * @param embeddingStore 向量存储
     * @param strategyName 策略名称
     * @return 包含各项生成指标的映射
     */
    public Map<String, Double> evaluateGeneration(EmbeddingStore<TextSegment> embeddingStore, String strategyName) {
        System.out.println("正在评估生成性能: " + strategyName);

        List<Double> faithfulnessScores = new ArrayList<>();      // 忠实度分数列表
        List<Double> hallucinationScores = new ArrayList<>();     // 幻觉率分数列表
        List<Double> contextUtilizationScores = new ArrayList<>(); // 上下文利用率分数列表

        // 对每个QA对进行生成测试
        for (QAPair qaPair : qaPairs) {
            @SuppressWarnings("unused")
            String referenceAnswer = qaPair.getReferenceAnswer(); // 预留用于未来评估改进
            Embedding query = embeddingModel.embed(qaPair.getQuestion()).content();
            EmbeddingSearchRequest embeddingSearchRequest = EmbeddingSearchRequest.builder()
                    .queryEmbedding(query)
                    .maxResults(10)//匹配最相似的10条记录
                    .minScore(0.0)
                    .build();
            // 检索相关文档作为上下文
            EmbeddingSearchResult<TextSegment> relevant = embeddingStore.search(embeddingSearchRequest);  // 检索top3相关文档
            List<EmbeddingMatch<TextSegment>> matches = relevant.matches();
            String context = matches.stream()
                    .map(embeddingMatch -> embeddingMatch.embedded().text())
                    .collect(Collectors.joining("\n"));

            // 构建提示词，要求模型基于上下文回答问题
            String prompt = String.format("""
                基于以下上下文回答问题：
                
                上下文：
                %s
                
                问题：%s
                
                请基于上下文提供准确回答：""", context, qaPair.getQuestion());

            try {

                // 使用LLM生成回答
                String generatedAnswer = chatModel.chat(prompt);

                // 评估生成回答的质量
                double faithfulness = evaluateFaithfulness(generatedAnswer, context);
                faithfulnessScores.add(faithfulness);

                double hallucination = evaluateHallucination(generatedAnswer, context);
                hallucinationScores.add(hallucination);

                double utilization = evaluateContextUtilization(generatedAnswer, context);
                contextUtilizationScores.add(utilization);

            } catch (Exception e) {
                System.out.println("生成评估错误: " + e.getMessage());
                // 发生错误时使用默认分数
                faithfulnessScores.add(0.0);
                hallucinationScores.add(1.0);
                contextUtilizationScores.add(0.0);
                continue;
            }
        }

        // 计算各项指标的平均值
        Map<String, Double> results = new HashMap<>();
        results.put("faithfulness", average(faithfulnessScores));
        results.put("hallucinationRate", average(hallucinationScores));
        results.put("contextUtilization", average(contextUtilizationScores));

        System.out.printf("生成评估完成: 忠实度=%.3f, 幻觉率=%.3f, 上下文利用率=%.3f%n",
                results.get("faithfulness"), results.get("hallucinationRate"),
                results.get("contextUtilization"));

        return results;
    }

    /**
     * 评估回答忠实度
     * 使用LLM进行语义级别的忠实度评估，比简单的单词重叠方法更准确
     * 该方法能够理解同义词、语义相似性和逻辑推断关系
     * @param answer 生成的回答
     * @param context 提供的上下文
     * @return 忠实度分数（0-1之间，越高表示越忠实）
     */
    private double evaluateFaithfulness(String answer, String context) {
        try {
            // 方法1: 使用LLM进行语义评估（推荐，更准确）
            return evaluateFaithfulnessWithLLM(answer, context);
        } catch (Exception e) {
            System.out.println("LLM忠实度评估失败，回退到基础方法: " + e.getMessage());
            // 方法2: 回退到改进的单词重叠方法
            return evaluateFaithfulnessWithEmbedding(answer, context);
        }
    }

    /**
     * 使用LLM进行忠实度评估（最准确的方法）
     * 通过让LLM判断答案是否可以从上下文中推断出来，进行语义级别的评估
     * @param answer 生成的回答
     * @param context 提供的上下文
     * @return 忠实度分数（0-1之间）
     */
    private double evaluateFaithfulnessWithLLM(String answer, String context) {
        String prompt = String.format("""
                请评估以下回答是否忠实于提供的上下文。
                
                上下文：
                %s
                
                回答：
                %s
                
                请判断：回答中的所有信息和事实是否都可以从上下文中直接或间接推断出来？
                如果回答完全基于上下文（包括可以合理推断的内容），输出1.0
                如果回答有部分信息无法从上下文中推断，请根据无法推断的比例输出0.0到1.0之间的分数
                如果回答完全不基于上下文，输出0.0
                
                只输出一个0到1之间的数字分数，不要输出其他内容：""", context, answer);

        try {
            String response = chatModel.chat(prompt).trim();
            
            // 尝试从响应中提取数字
            // LLM可能返回 "1.0" 或 "0.85" 或 "The score is 0.9" 等形式
            double score = extractScoreFromResponse(response);
            
            // 确保分数在有效范围内
            return Math.max(0.0, Math.min(1.0, score));
        } catch (Exception e) {
            throw new RuntimeException("LLM忠实度评估失败: " + e.getMessage(), e);
        }
    }

    /**
     * 使用Embedding语义相似度进行忠实度评估（中等准确度，比单词重叠更好）
     * 通过计算答案与上下文的语义相似度来评估忠实度
     * @param answer 生成的回答
     * @param context 提供的上下文
     * @return 忠实度分数（0-1之间）
     */
    private double evaluateFaithfulnessWithEmbedding(String answer, String context) {
        // 将答案分割成句子
        String[] answerSentences = answer.split("[。.!?\\n]");
        if (answerSentences.length == 0) return 0.0;

        // 为上下文生成embedding
        Embedding contextEmbedding = embeddingModel.embed(context).content();
        
        double totalScore = 0.0;
        int validSentences = 0;

        // 对答案的每个句子，计算与上下文的语义相似度
        for (String sentence : answerSentences) {
            sentence = sentence.trim();
            if (sentence.isEmpty() || sentence.length() < 3) continue;

            try {
                // 为答案句子生成embedding
                Embedding answerSentenceEmbedding = embeddingModel.embed(sentence).content();
                
                // 计算余弦相似度
                // 将float[]转换为List<Float>
                float[] contextVector = contextEmbedding.vector();
                float[] answerVector = answerSentenceEmbedding.vector();
                List<Float> contextVectorList = new ArrayList<>();
                List<Float> answerVectorList = new ArrayList<>();
                for (float f : contextVector) contextVectorList.add(f);
                for (float f : answerVector) answerVectorList.add(f);
                
                double similarity = cosineSimilarity(contextVectorList, answerVectorList);
                
                totalScore += similarity;
                validSentences++;
            } catch (Exception e) {
                // 如果某个句子处理失败，跳过它
                continue;
            }
        }

        if (validSentences == 0) return 0.0;
        
        // 返回平均相似度作为忠实度分数
        // 由于语义相似度可能较宽松，我们使用阈值调整
        double avgSimilarity = totalScore / validSentences;
        // 如果相似度很高（>0.7），认为忠实度较高
        // 使用sigmoid函数进行平滑映射
        return sigmoid(avgSimilarity * 2 - 1.0);
    }

    /**
     * 简单的单词重叠方法（快速但不够准确，作为最后备选）
     * @param answer 生成的回答
     * @param context 提供的上下文
     * @return 忠实度分数（0-1之间）
     */
    @SuppressWarnings("unused")
    private double evaluateFaithfulnessSimple(String answer, String context) {
        // 移除标点符号，转换为小写
        String normalizedAnswer = answer.toLowerCase().replaceAll("[^\\p{L}\\p{N}\\s]", "");
        String normalizedContext = context.toLowerCase().replaceAll("[^\\p{L}\\p{N}\\s]", "");
        
        // 转换为单词集合
        Set<String> answerWords = new HashSet<>(Arrays.asList(normalizedAnswer.split("\\s+")));
        Set<String> contextWords = new HashSet<>(Arrays.asList(normalizedContext.split("\\s+")));

        // 移除停用词（简化版，实际可以使用更完整的停用词列表）
        Set<String> stopWords = new HashSet<>(Arrays.asList(
            "的", "是", "在", "有", "和", "了", "就", "也", "与", "或", 
            "the", "is", "a", "an", "and", "or", "in", "on", "at", "to", "for", "of", "with"
        ));
        answerWords.removeAll(stopWords);
        contextWords.removeAll(stopWords);

        if (answerWords.isEmpty()) return 0.0;

        // 计算回答中出现在上下文中的单词比例
        long commonWords = answerWords.stream()
                .filter(word -> word.length() > 1 && contextWords.contains(word))
                .count();
        return (double) commonWords / answerWords.size();
    }

    /**
     * 从LLM响应中提取数字分数
     * @param response LLM的响应文本
     * @return 提取的分数，如果提取失败返回0.5
     */
    private double extractScoreFromResponse(String response) {
        // 尝试提取数字（可能包含小数）
        java.util.regex.Pattern pattern = java.util.regex.Pattern.compile("(\\d+\\.?\\d*)");
        java.util.regex.Matcher matcher = pattern.matcher(response);
        
        if (matcher.find()) {
            try {
                return Double.parseDouble(matcher.group(1));
            } catch (NumberFormatException e) {
                // 如果解析失败，返回默认值
                return 0.5;
            }
        }
        
        // 如果没有找到数字，根据关键词估算
        String lowerResponse = response.toLowerCase();
        if (lowerResponse.contains("完全") || lowerResponse.contains("fully") || lowerResponse.contains("100")) {
            return 1.0;
        } else if (lowerResponse.contains("完全不") || lowerResponse.contains("not at all") || lowerResponse.contains("0")) {
            return 0.0;
        } else if (lowerResponse.contains("部分") || lowerResponse.contains("partial")) {
            return 0.5;
        }
        
        return 0.5; // 默认值
    }

    /**
     * 计算两个向量的余弦相似度
     * @param vector1 第一个向量
     * @param vector2 第二个向量
     * @return 余弦相似度（-1到1之间）
     */
    private double cosineSimilarity(List<Float> vector1, List<Float> vector2) {
        if (vector1.size() != vector2.size()) {
            throw new IllegalArgumentException("向量维度不匹配");
        }

        double dotProduct = 0.0;
        double norm1 = 0.0;
        double norm2 = 0.0;

        for (int i = 0; i < vector1.size(); i++) {
            double v1 = vector1.get(i);
            double v2 = vector2.get(i);
            dotProduct += v1 * v2;
            norm1 += v1 * v1;
            norm2 += v2 * v2;
        }

        if (norm1 == 0.0 || norm2 == 0.0) {
            return 0.0;
        }

        return dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));
    }

    /**
     * Sigmoid函数，用于将相似度分数映射到更合理的范围
     * @param x 输入值
     * @return sigmoid映射后的值（0-1之间）
     */
    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    /**
     * 评估幻觉率
     * 衡量生成回答中包含不在上下文中信息的比例
     * @param answer 生成的回答
     * @param context 提供的上下文
     * @return 幻觉率分数（0-1之间，越高表示幻觉越多）
     */
    private double evaluateHallucination(String answer, String context) {
        // 幻觉率 = 1 - 忠实度
        return 1 - evaluateFaithfulness(answer, context);
    }

    /**
     * 评估上下文利用率
     * 使用语义相似度而非完全匹配来判断答案是否利用了上下文
     * @param answer 生成的回答
     * @param context 提供的上下文
     * @return 上下文利用率分数（0-1之间，越高表示利用越好）
     */
    private double evaluateContextUtilization(String answer, String context) {
        // 按句子分割回答和上下文
        String[] answerSentences = answer.split("[。.!?\\n]");
        String[] contextSentences = context.split("[。.!?\\n]");

        if (answerSentences.length == 0 || contextSentences.length == 0) return 0.0;

        // 为所有上下文句子生成embedding（缓存以避免重复计算）
        List<Embedding> contextEmbeddings = new ArrayList<>();
        for (String ctxSentence : contextSentences) {
            ctxSentence = ctxSentence.trim();
            if (ctxSentence.isEmpty() || ctxSentence.length() < 3) continue;
            try {
                Embedding embedding = embeddingModel.embed(ctxSentence).content();
                contextEmbeddings.add(embedding);
            } catch (Exception e) {
                // 如果某个上下文句子处理失败，跳过它
                continue;
            }
        }

        if (contextEmbeddings.isEmpty()) return 0.0;

        int utilizedCount = 0;
        int validAnswerSentences = 0;
        // 相似度阈值：超过此阈值的答案句子认为利用了上下文
        double similarityThreshold = 0.6;

        // 对每个答案句子，计算它与所有上下文句子的最大相似度
        for (String answerSentence : answerSentences) {
            answerSentence = answerSentence.trim();
            if (answerSentence.isEmpty() || answerSentence.length() < 3) continue;

            try {
                // 为答案句子生成embedding
                Embedding answerEmbedding = embeddingModel.embed(answerSentence).content();
                float[] answerVector = answerEmbedding.vector();
                List<Float> answerVectorList = new ArrayList<>();
                for (float f : answerVector) answerVectorList.add(f);

                // 计算与所有上下文句子的相似度，取最大值
                double maxSimilarity = 0.0;
                for (Embedding ctxEmbedding : contextEmbeddings) {
                    float[] contextVector = ctxEmbedding.vector();
                    List<Float> contextVectorList = new ArrayList<>();
                    for (float f : contextVector) contextVectorList.add(f);

                    double similarity = cosineSimilarity(answerVectorList, contextVectorList);
                    maxSimilarity = Math.max(maxSimilarity, similarity);
                }

                validAnswerSentences++;
                // 如果最大相似度超过阈值，认为该句子利用了上下文
                if (maxSimilarity >= similarityThreshold) {
                    utilizedCount++;
                }
            } catch (Exception e) {
                // 如果某个答案句子处理失败，跳过它
                continue;
            }
        }

        if (validAnswerSentences == 0) return 0.0;

        // 返回利用了上下文的句子比例
        return (double) utilizedCount / validAnswerSentences;
    }

    /**
     * 计算数值列表的平均值
     * @param values 数值列表
     * @return 平均值，如果列表为空则返回0
     */
    private double average(List<Double> values) {
        return values.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
    }


}
