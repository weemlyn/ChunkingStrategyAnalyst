package com.xwm.chunkingstrategyanalyst;

import dev.langchain4j.community.model.dashscope.QwenChatModel;
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;

@SpringBootTest
class ChunkingStrategyAnalystApplicationTests {

    @Test
    void contextLoads() {
        QwenChatModel model = QwenChatModel.builder()
                .modelName("qwen-flash")
                .apiKey("sk-196000a734cd4450a28ec874781bcdec")
                .build();
        String chat = model.chat("你是谁");
        System.out.println(chat);
    }

}
