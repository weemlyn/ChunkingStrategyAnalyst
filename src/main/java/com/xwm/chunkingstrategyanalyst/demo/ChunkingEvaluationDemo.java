package com.xwm.chunkingstrategyanalyst.demo;

import com.xwm.chunkingstrategyanalyst.entity.EvaluationMetrics;
import com.xwm.chunkingstrategyanalyst.entity.QAPair;
import com.xwm.chunkingstrategyanalyst.evaluator.ChunkingStrategyEvaluator;
import dev.langchain4j.data.document.Document;

import java.util.*;

/**
 * Chunking策略评估演示类
 * 展示如何使用评估器并进行结果分析
 */
public class ChunkingEvaluationDemo {

    /**
     * 主方法 - 程序入口
     */
    public static void main(String[] args) {
        System.out.println("=== RAG系统Chunking策略评估开始 ===");

        // 1. 准备测试数据
        List<Document> documents = prepareTestDocuments();
        List<QAPair> qaPairs = prepareTestQAPairs();

        System.out.printf("已准备 %d 个文档和 %d 个QA测试对%n", documents.size(), qaPairs.size());

        // 2. 创建评估器并运行评估
        ChunkingStrategyEvaluator evaluator = new ChunkingStrategyEvaluator(documents, qaPairs);
        List<EvaluationMetrics> results = evaluator.runComprehensiveEvaluation();

        // 3. 输出详细的评估结果
        printEvaluationResults(results);

        // 4. 生成策略推荐
        generateRecommendations(results);

        System.out.println("=== 评估完成 ===");
    }

    /**
     * 准备测试文档
     * 在实际应用中，这里可以加载真实的业务文档
     */
    private static List<Document> prepareTestDocuments() {
        List<Document> documents = Arrays.asList(
                // 文档1: 机器学习理论基础与技术体系
                Document.from("""
                    机器学习（Machine Learning, ML）作为人工智能（Artificial Intelligence, AI）的核心分支之一，
                    是一门研究如何使计算机系统通过经验自动改进性能的科学。与传统的基于规则的编程范式不同，
                    机器学习通过构建数学统计模型，使计算机能够从历史数据中识别复杂的模式，并利用这些模式
                    对新的未知数据进行预测或决策。机器学习的核心在于其自适应能力和泛化能力，即系统不仅能
                    在训练数据上表现良好，还能对未见过的数据做出合理的推断。
                    
                    从学习范式角度，机器学习主要分为三大类别：监督学习（Supervised Learning）、
                    无监督学习（Unsupervised Learning）和强化学习（Reinforcement Learning）。
                    监督学习依赖于标注数据集，通过学习输入特征到输出标签之间的映射关系来建立预测模型。
                    常见的监督学习任务包括分类（如垃圾邮件检测、图像识别）和回归（如房价预测、股票价格预测）。
                    典型的监督学习算法有线性回归、逻辑回归、决策树、随机森林、支持向量机（SVM）、
                    朴素贝叶斯以及近年来兴起的神经网络和深度学习模型。
                    
                    无监督学习则处理没有标签的数据，目标是发现数据中的隐藏结构、模式或聚类。
                    常见的无监督学习任务包括聚类分析（如客户细分、异常检测）、降维（如主成分分析PCA、
                    自编码器）、关联规则挖掘和生成模型。K-means聚类、层次聚类、DBSCAN是经典的聚类算法，
                    而PCA和t-SNE则广泛应用于数据可视化和特征提取。生成对抗网络（GAN）和变分自编码器（VAE）
                    是近年来在无监督生成领域取得突破的深度学习方法。
                    
                    强化学习模拟智能体（Agent）在环境（Environment）中通过试错学习最优策略的过程。
                    智能体通过执行动作（Action）并获得奖励（Reward）信号来学习在特定状态（State）下
                    应该采取何种行动以最大化长期累积奖励。强化学习在游戏AI（如AlphaGo、AlphaZero）、
                    自动驾驶、机器人控制、推荐系统优化等领域有广泛应用。Q-learning、策略梯度方法、
                    深度强化学习（DQN、PPO、A3C等）是该领域的代表性算法。
                    
                    机器学习的成功应用需要考虑多个关键因素：数据质量与数量、特征工程、模型选择与调优、
                    过拟合与欠拟合的平衡、模型解释性与可部署性。随着大数据和计算能力的发展，机器学习
                    技术已渗透到各个行业，从医疗诊断、金融风控到智能推荐、自然语言处理，都展现出巨大的应用潜力。
                    """),

                // 文档2: 深度学习架构与优化技术
                Document.from("""
                    深度学习（Deep Learning）是机器学习的一个专门分支，其核心思想是通过构建多层（深度）
                    人工神经网络（Artificial Neural Networks, ANN））来学习数据的层次化表示。
                    与传统机器学习方法依赖手工特征工程不同，深度学习能够自动从原始数据中学习到多层次、
                    抽象的特征表示，使得它在处理高维复杂数据时表现出色。深度学习的"深度"通常指网络包含
                    多个隐藏层（Hidden Layers），每层通过非线性激活函数（如ReLU、Sigmoid、Tanh）进行变换。
                    
                    卷积神经网络（Convolutional Neural Networks, CNN）是深度学习在计算机视觉领域最具影响力的架构。
                    CNN的核心思想是通过卷积操作（Convolution）和池化操作（Pooling）来提取图像的局部特征，
                    并通过多层卷积逐步构建更加抽象和全局的特征表示。卷积层通过可学习的卷积核（Kernel/Filter）
                    对输入特征图进行滑动窗口计算，捕获边缘、纹理、形状等低级特征，随后在更深层中组合形成
                    物体部件和完整对象的高级特征。经典的CNN架构包括LeNet-5、AlexNet、VGG、GoogLeNet、
                    ResNet（残差网络）、DenseNet等，其中ResNet通过跳跃连接（Skip Connection）解决了
                    深层网络的梯度消失问题，使得训练上百层的深度网络成为可能。
                    
                    循环神经网络（Recurrent Neural Networks, RNN）及其变体专门设计用于处理序列数据，
                    如文本、语音、时间序列等。RNN通过隐藏状态（Hidden State）在不同时间步之间传递信息，
                    使其能够捕捉序列中的时间依赖关系。然而，标准RNN存在梯度消失和梯度爆炸问题，难以学习
                    长期依赖。长短期记忆网络（LSTM）和门控循环单元（GRU）通过引入门控机制（Gate Mechanism）
                    有效解决了这一问题，成为序列建模的主流选择。LSTM使用遗忘门（Forget Gate）、输入门
                    （Input Gate）和输出门（Output Gate）来控制信息的流动，GRU则通过重置门和更新门
                    实现类似功能，且参数量更少、训练更快。
                    
                    Transformer架构的提出彻底改变了自然语言处理和序列建模领域。与RNN不同，Transformer
                    完全依赖注意力机制（Attention Mechanism）来建模序列依赖关系，摒弃了循环结构，
                    实现了并行计算，大大提升了训练效率。Transformer的核心是多头自注意力（Multi-Head
                    Self-Attention），它允许模型同时关注序列中不同位置的信息。位置编码（Positional
                    Encoding）被用来注入序列的顺序信息。Transformer架构包括编码器（Encoder）和解码器
                    （Decoder）两部分，编码器用于理解输入序列，解码器用于生成输出序列。
                    
                    基于Transformer的预训练语言模型已成为NLP的主流范式。BERT（Bidirectional Encoder
                    Representations from Transformers）采用双向编码器结构，通过掩码语言模型（MLM）
                    和下一句预测（NSP）任务进行预训练，在多个下游任务上刷新了最佳性能。GPT（Generative
                    Pre-trained Transformer）系列模型采用自回归（Autoregressive）的生成方式，
                    GPT-3和GPT-4展现了强大的少样本学习和指令遵循能力。T5（Text-to-Text Transfer
                    Transformer）将所有NLP任务统一为文本到文本的转换任务，简化了模型架构和训练流程。
                    
                    深度学习的成功离不开高效的训练技术。批量归一化（Batch Normalization）通过标准化
                    层输入加速训练并提高稳定性。Dropout通过随机失活部分神经元防止过拟合。学习率调度策略
                    （如余弦退火、Warm-up）帮助模型更好收敛。优化算法从SGD发展到Adam、AdamW、RMSprop等
                    自适应优化器。迁移学习（Transfer Learning）通过在大规模数据集上预训练模型，然后在
                    目标任务上微调，有效降低了数据需求和训练成本，这在计算机视觉和NLP领域都取得了显著效果。
                    """),

                // 文档3: 自然语言处理技术与应用生态
                Document.from("""
                    自然语言处理（Natural Language Processing, NLP）是计算机科学、人工智能和语言学
                    的交叉领域，旨在使计算机能够理解、解释、生成和操作人类语言。NLP的研究涵盖从低层的
                    文本预处理（如分词、词性标注、句法分析）到高层的语义理解、情感分析、机器翻译、
                    对话系统、文本摘要、问答系统等多个层面。
                    
                    词嵌入（Word Embedding）技术是NLP的基础技术之一，它将离散的词语映射到连续的
                    向量空间中，使得语义相似的词语在向量空间中距离较近。Word2Vec（包括Skip-gram和
                    CBOW两种模型）和GloVe（Global Vectors for Word Representation）是早期的经典方法，
                    它们通过无监督学习从大规模语料库中学习词向量。FastText通过引入子词（Subword）
                    信息，能够处理未登录词（OOV）问题，并更好地处理形态丰富的语言。
                    
                    Transformer架构的兴起推动了预训练语言模型（Pre-trained Language Models, PLM）
                    的快速发展。BERT及其变体（如RoBERTa、ALBERT、ELECTRA）通过双向上下文编码，
                    在理解任务上表现出色，广泛应用于文本分类、命名实体识别（NER）、关系抽取、情感分析等。
                    GPT系列模型基于自回归生成，在文本生成、对话系统、代码生成等任务上展现出卓越能力。
                    T5将各类NLP任务统一为序列到序列（Seq2Seq）任务，简化了模型设计。
                    
                    大语言模型（Large Language Models, LLM）如GPT-3.5、GPT-4、Claude、Gemini、
                    Llama等代表了NLP技术的最新进展。这些模型通常具有数千亿参数，通过指令微调
                    （Instruction Tuning）和人类反馈强化学习（RLHF）进行对齐，展现出了强大的
                    语言理解、推理、代码生成和多模态能力。Few-shot Learning和In-context Learning
                    使得这些模型能够在不更新参数的情况下通过提示（Prompt）完成新任务。
                    
                    RAG（Retrieval-Augmented Generation）架构结合了检索和生成的优势，通过从外部
                    知识库检索相关文档片段，然后基于这些片段生成回答，既提高了回答的准确性又增强了
                    可解释性。RAG系统通常包括文档切分（Chunking）、向量化嵌入、相似度检索和上下文
                    增强生成等步骤。Chunking策略的选择（如固定大小、语义分割、滑动窗口）直接影响
                    检索质量和生成效果。
                    
                    NLP的应用场景极其广泛。在机器翻译领域，神经网络机器翻译（NMT）已经超越传统
                    基于统计的方法，Google Translate、DeepL等系统实现了高质量的跨语言翻译。
                    在问答系统中，BERT等模型在SQuAD等数据集上接近人类水平。情感分析帮助企业
                    监控品牌声誉、分析用户反馈。命名实体识别和信息抽取从非结构化文本中提取结构化信息。
                    文本摘要技术可以生成新闻摘要、会议纪要。对话系统如ChatGPT、Claude等已广泛应用于
                    客服、教育、辅助创作等领域。
                    
                    NLP的发展仍面临诸多挑战：处理低资源语言、理解上下文中的隐含信息、处理长文本、
                    保证生成内容的准确性和安全性、跨语言迁移、多模态理解（结合图像、音频）等。
                    随着模型规模的增长，计算成本、能耗和环境影响也成为需要考虑的重要因素。
                    模型的可解释性、公平性和偏见问题也需要持续关注和改进。
                    """),

                // 文档4: 机器学习在工业界的实践与挑战
                Document.from("""
                    将机器学习从实验室原型转化为生产环境中的可靠系统，面临着诸多工程和实践挑战。
                    MLOps（Machine Learning Operations）已成为连接数据科学家和软件工程师的重要桥梁，
                    涵盖了模型开发、训练、部署、监控和迭代优化的全生命周期管理。
                    
                    数据质量是决定ML系统成功的关键因素。常见的数据质量问题包括：数据缺失、重复记录、
                    标注错误、数据分布偏移、类别不平衡等。数据预处理流程通常包括数据清洗、特征工程、
                    特征选择、数据增强和特征标准化。特征工程是提升模型性能的重要手段，包括特征变换
                    （如对数变换、独热编码）、特征组合（如多项式特征）、时间特征提取、文本特征提取等。
                    在深度学习中，虽然自动特征学习能力增强，但在某些领域，精心设计的特征仍能带来
                    显著提升。
                    
                    模型训练过程中的超参数调优是一个重要但耗时的工作。网格搜索（Grid Search）、
                    随机搜索（Random Search）、贝叶斯优化（Bayesian Optimization）是常见的超参数
                    优化方法。交叉验证（Cross-validation）用于评估模型的泛化能力，K折交叉验证能够
                    更充分地利用有限的数据。早停（Early Stopping）机制可以防止过拟合，正则化技术
                    （L1、L2正则化、Dropout）也在模型训练中广泛应用。
                    
                    模型评估指标的选择取决于具体任务和业务目标。对于分类任务，准确率（Accuracy）、
                    精确率（Precision）、召回率（Recall）、F1分数、AUC-ROC曲线是常用指标。
                    对于回归任务，均方误差（MSE）、平均绝对误差（MAE）、R²决定系数是常见评估标准。
                    在多类别分类中，混淆矩阵（Confusion Matrix）和分类报告提供了详细的性能分析。
                    在推荐系统中，NDCG（Normalized Discounted Cumulative Gain）、MAP
                    （Mean Average Precision）等排序指标更为适用。
                    
                    模型部署时需要考虑延迟、吞吐量、资源消耗、可扩展性等要求。模型压缩技术
                    （如量化、剪枝、知识蒸馏）可以在保持性能的同时减小模型大小和推理时间。
                    边缘计算和模型服务化（Model Serving）使得模型能够快速响应实时请求。
                    A/B测试是验证模型在生产环境中效果的关键方法，通过对比新旧模型的实际表现来决定
                    是否上线新模型。
                    
                    模型监控和持续改进是MLOps的重要环节。数据漂移（Data Drift）和概念漂移
                    （Concept Drift）可能导致模型性能下降，需要建立监控机制及时检测并触发模型
                    重训练。模型解释性技术（如SHAP、LIME）帮助理解模型的决策过程，增强用户信任。
                    在多模型系统中，模型版本管理、特征存储、实验追踪等工具链对提高开发效率至关重要。
                    """),

                // 文档5: 神经网络架构演进与优化算法
                Document.from("""
                    神经网络的发展历程可以追溯到20世纪40年代的感知机（Perceptron），但真正的突破
                    发生在21世纪初，特别是2012年AlexNet在ImageNet竞赛中的成功，标志着深度学习时代的
                    到来。自那以后，神经网络架构不断演进，涌现出众多创新设计。
                    
                    残差网络（ResNet）通过引入残差连接（Residual Connection）解决了深层网络的
                    退化问题，使得训练数百甚至上千层的网络成为可能。残差连接允许信息直接从某一层
                    传递到后续层，缓解了梯度消失问题，并使得网络能够学习恒等映射。ResNet的成功启发
                    了后续架构设计，如DenseNet中的密集连接（Dense Connection）进一步增强了特征的
                    流动和重用。
                    
                    注意力机制最早在机器翻译任务中提出，允许模型动态地关注输入序列的不同部分。
                    自注意力（Self-Attention）扩展了这一概念，使得序列中的每个位置都能够关注到
                    序列中的所有其他位置，包括自身。多头注意力（Multi-Head Attention）通过并行
                    运行多个注意力机制，使得模型能够同时关注不同类型的信息。注意力机制的计算效率
                    和并行化优势，使其成为Transformer架构的核心，并广泛应用到视觉、语音等领域。
                    
                    图神经网络（Graph Neural Networks, GNN）专门设计用于处理图结构数据，如社交网络、
                    分子结构、知识图谱等。GNN通过消息传递机制更新节点表示，使得每个节点能够聚合
                    来自邻居节点的信息。GCN（Graph Convolutional Network）、GAT（Graph Attention
                    Network）、GraphSAGE等是代表性架构。GNN在推荐系统、药物发现、交通预测等
                    领域展现出强大能力。
                    
                    优化算法对神经网络的训练至关重要。随机梯度下降（SGD）虽然简单，但收敛较慢且
                    对学习率敏感。动量法（Momentum）通过引入历史梯度信息，加速收敛并减少震荡。
                    Adam（Adaptive Moment Estimation）结合了动量和自适应学习率的优点，能够为
                    每个参数动态调整学习率，是当前最常用的优化器之一。AdamW通过在权重衰减上的改进，
                    进一步提升了泛化性能。学习率调度策略如Warm-up、Cosine Annealing、
                    OneCycleLR等也对训练效果有显著影响。
                    
                    正则化技术用于防止过拟合并提升模型泛化能力。除传统的L1、L2正则化外，Dropout
                    随机丢弃神经元、DropConnect随机断开连接、权重衰减（Weight Decay）都是常用方法。
                    批量归一化（Batch Normalization）通过标准化层输入，不仅起到正则化作用，还加速了
                    训练。层归一化（Layer Normalization）和组归一化（Group Normalization）是
                    Batch Normalization的变体，适用于不同场景。
                    """),

                // 文档6: 计算机视觉中的深度学习应用
                Document.from("""
                    计算机视觉是深度学习的另一个重要应用领域，深度学习技术已经在图像分类、目标检测、
                    语义分割、人脸识别、图像生成等任务上取得了突破性进展。
                    
                    图像分类是最基础的视觉任务，CNN在此领域的成功奠定了深度学习的地位。从LeNet到
                    AlexNet、VGG、GoogLeNet、ResNet、EfficientNet，模型架构不断演进，在ImageNet
                    数据集上的Top-5错误率从26%降低到2%以下，甚至超越人类水平。EfficientNet通过
                    平衡网络的深度、宽度和分辨率，在保持准确率的同时大幅降低了参数量和计算量。
                    
                    目标检测不仅要识别图像中的物体类别，还要定位物体的位置。两阶段检测器如R-CNN、
                    Fast R-CNN、Faster R-CNN通过先生成候选区域再分类的思路，准确率较高但速度较慢。
                    单阶段检测器如YOLO（You Only Look Once）、SSD（Single Shot MultiBox Detector）
                    直接在图像上预测边界框和类别，速度更快但精度略低。YOLO的最新版本（如YOLOv8、
                    YOLOv9）通过改进网络架构和训练策略，在速度和精度上取得了更好的平衡。
                    
                    语义分割将图像中的每个像素分配到对应的类别，常用架构包括FCN（Fully Convolutional
                    Network）、U-Net、DeepLab、SegFormer等。U-Net通过U形的编码器-解码器结构和
                    跳跃连接，在医学图像分割等领域表现优异。DeepLab通过空洞卷积（Atrous/Dilated
                    Convolution）和金字塔池化模块，能够捕获多尺度上下文信息。
                    
                    人脸识别技术在安防、支付、社交等场景广泛应用。深度学习方法如FaceNet、
                    ArcFace通过度量学习（Metric Learning）优化人脸特征表示，使得同一人的不同
                    人脸图像在特征空间中距离较近，不同人的距离较远。Triplet Loss、Center Loss、
                    ArcFace Loss等损失函数设计对提升识别准确率至关重要。
                    
                    生成对抗网络（GAN）和扩散模型（Diffusion Models）是图像生成领域的两大主流技术。
                    GAN通过生成器和判别器的对抗训练，能够生成高质量的逼真图像。StyleGAN、StyleGAN2
                    在生成质量和可控性上取得了突破。扩散模型通过逐步去噪的过程生成图像，在DALL·E 2、
                    Stable Diffusion、Midjourney等系统中展现出惊人的生成能力。
                    
                    视觉Transformer（Vision Transformer, ViT）将Transformer架构引入计算机视觉领域，
                    通过将图像切分成固定大小的图像块（Patch）并编码为序列，将图像分类转化为序列分类
                    问题。ViT在大规模数据集上预训练后，能够达到甚至超越CNN的性能。Swin Transformer
                    通过窗口划分和移位窗口机制，使ViT能够处理不同分辨率的图像，并在目标检测和
                    分割任务上取得了优秀成果。
                    """)
        );

        // 为文档添加元数据（在实际应用中可能包含文档ID、来源等信息）
        for (int i = 0; i < documents.size(); i++) {
            // 这里简化处理，实际可以使用更复杂的元数据
        }

        return documents;
    }

    /**
     * 准备QA测试对
     * 每个QA对包含问题、相关文档列表和参考答案
     */
    private static List<QAPair> prepareTestQAPairs() {
        List<QAPair> qaPairs = Arrays.asList(
                // QA对1: 关于机器学习三个主要范式的详细问题（需要深入理解文档1）
                new QAPair(
                        "请详细解释机器学习的三大学习范式，并说明每种范式的特点、典型算法和应用场景？",
                        Arrays.asList("doc1"),
                        "机器学习的三大范式包括：1) 监督学习：依赖标注数据，学习输入特征到输出标签的映射，"
                        + "典型算法有线性回归、逻辑回归、决策树、随机森林、SVM、神经网络，应用于分类和回归任务如"
                        + "垃圾邮件检测、图像识别、房价预测；2) 无监督学习：处理无标签数据，发现隐藏结构和模式，"
                        + "包括聚类（K-means、层次聚类、DBSCAN）、降维（PCA、t-SNE）、生成模型（GAN、VAE），"
                        + "应用于客户细分、异常检测、数据可视化；3) 强化学习：智能体在环境中通过试错学习最优策略，"
                        + "算法如Q-learning、策略梯度方法、深度强化学习（DQN、PPO、A3C），应用于游戏AI、"
                        + "自动驾驶、机器人控制、推荐系统优化。"
                ),

                // QA对2: 关于Transformer架构变革性影响的问题（需要深入理解文档2）
                new QAPair(
                        "Transformer架构相比RNN有哪些根本性改进？它的核心组件是什么？基于Transformer的预训练模型有哪些代表性工作？",
                        Arrays.asList("doc2"),
                        "Transformer相比RNN的根本改进：1) 完全摒弃循环结构，依赖注意力机制建模序列依赖，"
                        + "实现了并行计算，大幅提升训练效率；2) 通过多头自注意力机制同时关注序列中不同位置的信息；"
                        + "3) 引入位置编码注入序列顺序信息。核心组件包括：多头自注意力（Multi-Head Self-Attention）、"
                        + "位置编码（Positional Encoding）、编码器-解码器架构。基于Transformer的代表性预训练模型："
                        + "BERT（双向编码器，通过MLM和NSP任务预训练）、GPT系列（自回归生成，展现少样本学习能力）、"
                        + "T5（将所有NLP任务统一为文本到文本的转换任务）。"
                ),

                // QA对3: 关于RAG系统与NLP应用的综合问题（需要深入理解文档3）
                new QAPair(
                        "什么是RAG架构？它如何结合检索和生成？RAG系统中的Chunking策略对系统性能有什么影响？大语言模型在NLP领域的应用体现在哪些方面？",
                        Arrays.asList("doc3"),
                        "RAG（Retrieval-Augmented Generation）架构结合了检索和生成的优势，通过从外部知识库"
                        + "检索相关文档片段，然后基于这些片段生成回答，既提高了回答准确性又增强了可解释性。"
                        + "RAG系统包括：文档切分（Chunking）、向量化嵌入、相似度检索和上下文增强生成等步骤。"
                        + "Chunking策略的选择（如固定大小、语义分割、滑动窗口）直接影响检索质量和生成效果。"
                        + "大语言模型（LLM）如GPT-3.5、GPT-4、Claude、Gemini等在NLP领域展现出强大的语言理解、"
                        + "推理、代码生成和多模态能力，通过指令微调和RLHF进行对齐，Few-shot Learning和"
                        + "In-context Learning使得模型能够通过提示完成新任务，广泛应用于对话系统、客服、教育、辅助创作等领域。"
                ),

                // QA对4: 跨文档综合问题 - 机器学习和深度学习的演进关系（涉及doc1和doc2）
                new QAPair(
                        "从机器学习到深度学习的演进过程中，关键技术突破有哪些？深度学习的自动特征学习能力相比传统机器学习有什么优势？",
                        Arrays.asList("doc1", "doc2"),
                        "关键演进：1) 从依赖手工特征工程的传统ML到自动学习特征的深度学习；2) CNN通过卷积和池化"
                        + "自动提取图像的多层次特征，在计算机视觉领域取得突破；3) RNN/LSTM/GRU通过隐藏状态传递"
                        + "捕获序列依赖，但存在梯度问题；4) Transformer通过注意力机制完全摒弃循环，实现并行计算，"
                        + "彻底改变NLP；5) 残差连接（ResNet）解决深层网络退化问题；6) 批量归一化、Dropout等训练技术"
                        + "加速训练并防止过拟合；7) Adam等自适应优化器提升训练效率；8) 迁移学习降低数据需求和成本。"
                        + "深度学习的优势：能够从原始数据自动学习多层次抽象特征表示，无需手工设计特征，在高维复杂数据"
                        + "处理时表现出色，通过端到端学习优化整个系统性能。"
                ),

                // QA对5: 关于神经网络训练优化的综合问题（涉及doc2和doc5）
                new QAPair(
                        "深度神经网络训练过程中常用的优化算法有哪些？它们各自的优缺点是什么？正则化技术如何防止过拟合？",
                        Arrays.asList("doc2", "doc5"),
                        "优化算法演进：1) SGD：简单但对学习率敏感、收敛慢；2) 动量法：引入历史梯度信息，"
                        + "加速收敛并减少震荡；3) Adam/AdamW：结合动量和自适应学习率，为每个参数动态调整学习率，"
                        + "是当前最常用的优化器，AdamW在权重衰减上改进提升了泛化性能。学习率调度策略如Warm-up、"
                        + "Cosine Annealing、OneCycleLR也对训练效果有显著影响。正则化技术：1) L1/L2正则化："
                        + "约束模型参数；2) Dropout/DropConnect：随机失活神经元或连接，防止过拟合；3) 权重衰减："
                        + "对参数进行惩罚；4) 批量归一化：标准化层输入，起到正则化作用并加速训练；5) 层归一化和"
                        + "组归一化是Batch Normalization的变体。"
                ),

                // QA对6: 关于MLOps和生产部署的实践问题（涉及doc4）
                new QAPair(
                        "MLOps在机器学习系统生命周期中起什么作用？从模型训练到生产部署需要关注哪些关键环节？如何处理数据漂移和概念漂移问题？",
                        Arrays.asList("doc4"),
                        "MLOps连接数据科学家和软件工程师，涵盖模型开发、训练、部署、监控和迭代优化的全生命周期。"
                        + "关键环节：1) 数据质量保障（数据清洗、特征工程、特征选择、数据增强）；2) 超参数调优"
                        + "（网格搜索、随机搜索、贝叶斯优化）和交叉验证；3) 模型评估指标选择（分类：Accuracy、"
                        + "Precision、Recall、F1、AUC-ROC；回归：MSE、MAE、R²；推荐：NDCG、MAP）；4) 模型部署"
                        + "（考虑延迟、吞吐量、资源消耗、可扩展性，使用模型压缩技术如量化、剪枝、知识蒸馏）；"
                        + "5) A/B测试验证生产效果；6) 模型监控（数据漂移、概念漂移检测，及时触发重训练）；"
                        + "7) 模型解释性技术（SHAP、LIME）增强用户信任；8) 模型版本管理、特征存储、实验追踪等工具链。"
                ),

                // QA对7: 跨文档综合问题 - 注意力机制的应用（涉及doc2和doc5）
                new QAPair(
                        "注意力机制是如何发展的？它在Transformer和传统RNN中的实现有什么不同？注意力机制还被应用到了哪些领域？",
                        Arrays.asList("doc2", "doc5"),
                        "注意力机制发展：最初在机器翻译任务中提出，允许模型动态关注输入序列的不同部分。"
                        + "在Transformer中的创新：1) 自注意力（Self-Attention）使序列中每个位置都能关注到"
                        + "所有其他位置包括自身；2) 多头注意力并行运行多个注意力机制，同时关注不同类型信息；"
                        + "3) 计算效率和并行化优势使其成为Transformer核心。与RNN的区别：RNN通过循环结构逐 step"
                        + "处理序列，难以并行；Transformer完全依赖注意力，无循环结构，可并行计算。应用扩展："
                        + "从NLP扩展到计算机视觉（Vision Transformer、Swin Transformer）、语音识别、图神经网络"
                        + "（GAT - Graph Attention Network）等多个领域，展现出强大的泛化能力。"
                ),

                // QA对8: 关于计算机视觉和NLP交叉应用的问题（涉及doc3和doc6）
                new QAPair(
                        "Transformer架构是如何从NLP领域迁移到计算机视觉领域的？ViT和传统CNN相比有什么优势和局限性？在视觉任务中，Transformer和CNN能否结合使用？",
                        Arrays.asList("doc3", "doc6"),
                        "Transformer在计算机视觉的迁移：Vision Transformer（ViT）将图像切分成固定大小的图像块"
                        + "（Patch）并编码为序列，将图像分类转化为序列分类问题，在大规模数据集预训练后达到甚至超越"
                        + "CNN性能。Swin Transformer通过窗口划分和移位窗口机制，使ViT能处理不同分辨率图像，"
                        + "在目标检测和分割任务上取得优秀成果。ViT优势：1) 并行计算能力强；2) 长距离依赖建模能力"
                        + "优于CNN；3) 在大规模数据上预训练后泛化能力强。局限性：1) 需要大量训练数据；2) 计算资源"
                        + "消耗较大；3) 在小数据集上可能不如CNN。结合使用：可以结合两者优势，如使用CNN作为特征提取器，"
                        + "然后接Transformer进行序列建模，或使用混合架构如ConViT、DeiT等。"
                ),

                // QA对9: 关于深度学习在不同领域应用的跨领域问题（涉及doc2、doc3、doc5、doc6）
                new QAPair(
                        "深度学习在计算机视觉、自然语言处理、图数据处理等领域分别使用了哪些核心架构？这些架构是如何针对各自领域特点设计的？",
                        Arrays.asList("doc2", "doc3", "doc5", "doc6"),
                        "各领域核心架构：1) 计算机视觉：CNN通过卷积和池化提取局部特征，构建层次化表示（LeNet、"
                        + "AlexNet、VGG、ResNet、DenseNet、EfficientNet）；Transformer通过ViT和Swin Transformer"
                        + "实现图像分类、检测、分割；2) 自然语言处理：早期RNN/LSTM/GRU处理序列，Transformer成为"
                        + "主流，BERT用于理解任务，GPT用于生成任务，大语言模型展现强大能力；3) 图数据处理："
                        + "GNN通过消息传递机制更新节点表示，GCN、GAT、GraphSAGE在推荐系统、药物发现、交通预测"
                        + "等领域应用。设计特点：CNN针对图像的空间局部性和平移不变性；RNN/Transformer针对序列的"
                        + "时间/顺序依赖；GNN针对图的拓扑结构和节点关系。"
                ),

                // QA对10: 关于强化学习及其与其他学习范式关系的深入问题（涉及doc1，需要推理）
                new QAPair(
                        "强化学习与监督学习和无监督学习有什么根本区别？强化学习中的奖励信号起什么作用？深度强化学习如何结合深度学习和强化学习？",
                        Arrays.asList("doc1", "doc2"),
                        "根本区别：1) 监督学习依赖标注数据，学习输入到输出的映射；无监督学习发现数据中的隐藏结构；"
                        + "强化学习通过智能体与环境的交互学习，没有预先给定的正确答案，而是通过奖励信号学习最优策略。"
                        + "2) 强化学习是顺序决策问题，当前决策影响未来状态和奖励；监督/无监督学习通常是独立样本学习。"
                        + "奖励信号作用：1) 提供学习信号，引导智能体朝向目标行为；2) 设计良好的奖励函数至关重要；"
                        + "3) 智能体通过最大化长期累积奖励学习策略；4) 奖励可能是稀疏的（只在特定时刻给出）或密集的。"
                        + "深度强化学习结合：使用深度神经网络（如CNN、全连接网络）作为值函数近似器（如DQN）或策略网络"
                        + "（如Actor-Critic方法PPO、A3C），使强化学习能够处理高维状态空间和动作空间，在Atari游戏、"
                        + "AlphaGo等任务上取得突破。"
                )
        );

        return qaPairs;
    }

    /**
     * 打印详细的评估结果
     * 以表格形式展示各种策略的各项指标
     */
    private static void printEvaluationResults(List<EvaluationMetrics> results) {
        System.out.println("\n" + "=".repeat(100));
        System.out.println("CHUNKING策略评估详细结果");
        System.out.println("=".repeat(100));

        // 表头
        System.out.printf("%-20s %-10s %-12s %-8s %-10s %-10s %-12s %-10s%n",
                "策略", "Recall@5", "Precision@5", "MRR", "忠实度", "幻觉率", "构建时间(s)", "索引大小");
        System.out.println("-".repeat(100));

        // 逐行输出每个策略的结果
        for (EvaluationMetrics metrics : results) {
            System.out.printf("%-20s %-10.3f %-12.3f %-8.3f %-10.3f %-10.3f %-12.2f %-10d%n",
                    metrics.getStrategyName(),
                    metrics.getRecallAtK().get(5),
                    metrics.getPrecisionAtK().get(5),
                    metrics.getMrr(),
                    metrics.getFaithfulness(),
                    metrics.getHallucinationRate(),
                    metrics.getBuildSpeed(),
                    metrics.getIndexSize());
        }

        System.out.println("=".repeat(100));
    }

    /**
     * 生成策略推荐
     * 基于评估结果给出针对不同场景的最佳策略建议
     */
    private static void generateRecommendations(List<EvaluationMetrics> results) {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("策略推荐分析");
        System.out.println("=".repeat(80));

        // 找出在不同指标上表现最好的策略
        Optional<EvaluationMetrics> bestRecall = results.stream()
                .max(Comparator.comparing(m -> m.getRecallAtK().get(5)));

        Optional<EvaluationMetrics> bestPrecision = results.stream()
                .max(Comparator.comparing(m -> m.getPrecisionAtK().get(5)));

        Optional<EvaluationMetrics> bestFaithfulness = results.stream()
                .max(Comparator.comparing(EvaluationMetrics::getFaithfulness));

        Optional<EvaluationMetrics> fastestBuild = results.stream()
                .min(Comparator.comparing(EvaluationMetrics::getBuildSpeed));

        Optional<EvaluationMetrics> smallestIndex = results.stream()
                .min(Comparator.comparing(EvaluationMetrics::getIndexSize));

        // 输出最佳策略
        System.out.println("基于评估结果的策略推荐:");
        System.out.println();

        bestRecall.ifPresent(metrics ->
                System.out.printf("🏆 最佳召回率策略: %-20s (Recall@5: %.3f)%n",
                        metrics.getStrategyName(), metrics.getRecallAtK().get(5)));

        bestPrecision.ifPresent(metrics ->
                System.out.printf("🎯 最佳精确率策略: %-20s (Precision@5: %.3f)%n",
                        metrics.getStrategyName(), metrics.getPrecisionAtK().get(5)));

        bestFaithfulness.ifPresent(metrics ->
                System.out.printf("✅ 最佳忠实度策略: %-20s (忠实度: %.3f)%n",
                        metrics.getStrategyName(), metrics.getFaithfulness()));

        fastestBuild.ifPresent(metrics ->
                System.out.printf("⚡ 最快构建策略: %-20s (构建时间: %.2fs)%n",
                        metrics.getStrategyName(), metrics.getBuildSpeed()));

        smallestIndex.ifPresent(metrics ->
                System.out.printf("💾 最小索引策略: %-20s (索引大小: %d)%n",
                        metrics.getStrategyName(), metrics.getIndexSize()));

        System.out.println();
        System.out.println("场景化推荐:");
        System.out.println("• 📈 重检索精度场景: " + bestRecall.map(EvaluationMetrics::getStrategyName).orElse("N/A"));
        System.out.println("• 🤖 重生成质量场景: " + bestFaithfulness.map(EvaluationMetrics::getStrategyName).orElse("N/A"));
        System.out.println("• 🚀 重响应速度场景: " + fastestBuild.map(EvaluationMetrics::getStrategyName).orElse("N/A"));
        System.out.println("• 💰 重资源效率场景: " + smallestIndex.map(EvaluationMetrics::getStrategyName).orElse("N/A"));

        System.out.println();
        System.out.println("💡 提示: 实际选择时应根据具体业务需求权衡各项指标");
    }
}
