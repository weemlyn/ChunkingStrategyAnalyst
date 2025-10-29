package com.xwm.chunkingstrategyanalyst.entity;

import java.util.Map;

public class EvaluationMetrics {
    private final String strategyName;
    private final Map<Integer, Double> recallAtK;
    private final Map<Integer, Double> precisionAtK;
    private final double mrr;
    private final double ndcg;
    private final double latency;
    private final int indexSize;
    private final double faithfulness;
    private final double contextUtilization;
    private final double hallucinationRate;
    private final double buildSpeed;

    /**
     * 构造函数
     * @param strategyName 策略名称
     * @param recallAtK 不同K值的召回率
     * @param precisionAtK 不同K值的精确率
     * @param mrr 平均倒数排名
     * @param ndcg 归一化折损累积增益
     * @param latency 检索延迟
     * @param indexSize 索引大小
     * @param faithfulness 忠实度
     * @param contextUtilization 上下文利用率
     * @param hallucinationRate 幻觉率
     * @param buildSpeed 构建速度
     */
    public EvaluationMetrics(String strategyName, Map<Integer, Double> recallAtK,
                             Map<Integer, Double> precisionAtK, double mrr, double ndcg,
                             double latency, int indexSize, double faithfulness,
                             double contextUtilization, double hallucinationRate, double buildSpeed) {
        this.strategyName = strategyName;
        this.recallAtK = recallAtK;
        this.precisionAtK = precisionAtK;
        this.mrr = mrr;
        this.ndcg = ndcg;
        this.latency = latency;
        this.indexSize = indexSize;
        this.faithfulness = faithfulness;
        this.contextUtilization = contextUtilization;
        this.hallucinationRate = hallucinationRate;
        this.buildSpeed = buildSpeed;
    }

    // Getter methods...
    public String getStrategyName() { return strategyName; }
    public Map<Integer, Double> getRecallAtK() { return recallAtK; }
    public Map<Integer, Double> getPrecisionAtK() { return precisionAtK; }
    public double getMrr() { return mrr; }
    public double getNdcg() { return ndcg; }
    public double getLatency() { return latency; }
    public int getIndexSize() { return indexSize; }
    public double getFaithfulness() { return faithfulness; }
    public double getContextUtilization() { return contextUtilization; }
    public double getHallucinationRate() { return hallucinationRate; }
    public double getBuildSpeed() { return buildSpeed; }

    @Override
    public String toString() {
        return String.format(
                "EvaluationMetrics{strategy='%s', recall@5=%.3f, precision@5=%.3f, faithfulness=%.3f}",
                strategyName, recallAtK.get(5), precisionAtK.get(5), faithfulness);
    }
}
