/* student.c - Lab13 学员要改的文件
 *
 * 三个 TODO 函数 + 一个主循环 = 把 12 章的 BPE / 模型 / KV Cache / 采样缝起来。
 *
 * 重要约束:
 *   - 不要 include "../framework/*.c"——所有 framework 代码由 Makefile 链接
 *   - 不要改 framework/ 任何 .c——它们是"标准答案"
 *   - 写完后跑 make clean && make test，对照 TASK.md 里的真实目标输出检查结果
 */

#include "student.h"
#include "verify.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bpe_tokenizer.h"   /* BPETokenizer / bpe_encode / bpe_decode */
#include "model.h"           /* GPTModel / model_forward / model_cache_* */
#include "tensor.h"          /* Tensor / tensor_zeros / tensor_free */
#include "config.h"          /* ModelConfig / tiny_config */
#include "generate.h"        /* sample_top_k / sample_greedy */

static char* dup_cstr(const char* s) {
    size_t len = strlen(s) + 1;
    char* out = (char*)malloc(len);
    if (out != NULL) memcpy(out, s, len);
    return out;
}

/* ============================================================
 * 任务 1.5.1：BPE 编码用户输入
 * ============================================================ */
int* student_bpe_chat_tokenize(BPETokenizer* tok,
                               const char* user_input,
                               int* out_len) {
    /* TODO(student):
     *   1. NULL 守护：tok / user_input / out_len 任一为 NULL 时返回 NULL
     *   2. 调 bpe_encode(tok, user_input, out_len, 1, 0)  // add_bos=1, add_eos=0
     *   3. bpe_encode 内部会 malloc 出 ID 数组，直接 return 它
     *   4. 别忘了把 *out_len 填好（bpe_encode 自己会填，但你要确认）
     *
     * 参考：framework/bpe_tokenizer.h 第 99-100 行的 bpe_encode 签名
     */
    (void)tok;
    (void)user_input;
    (void)out_len;
    return NULL;   /* 占位 return，学员替换成 bpe_encode 的返回值 */
}

/* ============================================================
 * 任务 1.5.2：BPE 解码助手回复
 * ============================================================ */
char* student_bpe_chat_decode(BPETokenizer* tok,
                              int* token_ids,
                              int len) {
    /* TODO(student):
     *   1. NULL 守护：tok / token_ids 为 NULL 或 len <= 0 时返回 NULL
     *   2. 调 bpe_decode(tok, token_ids, len) 拿到堆上的字符串
     *   3. return 它
     */
    (void)tok;
    (void)token_ids;
    (void)len;
    return NULL;   /* 占位 return */
}

/* ============================================================
 * 任务 1.5.3：单轮对话（核心题）
 * ============================================================ */
char* student_bpe_chat_one_turn(GPTModel* model,
                                BPETokenizer* tok,
                                GPTCache* cache,
                                const char* user_input,
                                int max_new_tokens) {
    /* TODO(student):
     *   1. NULL 守护：model / tok / cache / user_input 任一为 NULL 时返回 NULL
     *   2. prompt_ids = student_bpe_chat_tokenize(tok, user_input, &prompt_len)
     *   3. 申请 logits = tensor_zeros(2, (int[]){prompt_len, vocab_size})
     *      其中 vocab_size = model->config.vocab_size
     *   4. 申请 last_logits = tensor_zeros(1, (int[]){vocab_size})
     *   5. model_forward(model, prompt_ids, prompt_len, cache, logits)
     *   6. 把 logits 最后一行（[vocab_size]）拷到 last_logits
     *      —— 简单实现：循环 vocab_size 次 tensor_set / tensor_get
     *      —— 或者用 memcpy，前提是 logits->data 末尾就是最后一行
     *   7. 申请 out_ids = malloc(max_new_tokens * sizeof(int))
     *   8. 循环 max_new_tokens 次:
     *        - new_id = sample_top_k(last_logits, 1, 1.0f)  // k=1 等价贪心
     *        - out_ids[i] = new_id
     *        - 如果 new_id == tok->eos_id 跳出
     *        - 把 out_ids[i] 作为下一次 model_forward 的 1-token 输入
     *        - 拿新一轮 logits 的最后一行
     *   9. char* text = student_bpe_chat_decode(tok, out_ids, actual_len)
     *  10. 收尾：free(prompt_ids) / free(out_ids) / tensor_free(logits) / tensor_free(last_logits)
     *  11. return text
     *
     * 提示：第 8 步第二次 model_forward 的输入长度 = 1（只喂刚采的 token）
     *      ——这是 KV Cache 复用的关键。
     */
    (void)model;
    (void)tok;
    (void)cache;
    (void)user_input;
    (void)max_new_tokens;
    return dup_cstr("(TODO)");   /* 占位返回 */
}

/* ============================================================
 * 任务 1.5.4：主循环
 * ============================================================ */
void student_chat_loop(GPTModel* model,
                       BPETokenizer* tok,
                       GPTCache* cache) {
    /* TODO(student):
     *   1. 打印欢迎语（"BPE Chat (type /quit to exit)"）
     *   2. char buf[1024]
     *   3. 无限循环:
     *        - printf("> "); fflush(stdout);
     *        - if (fgets(buf, sizeof(buf), stdin) == NULL) break;  // EOF
     *        - 去掉 buf 末尾的 '\n'
     *        - if (strcmp(buf, "/quit") == 0) break;
     *        - reply = student_bpe_chat_one_turn(model, tok, cache, buf, 32);
     *        - if (reply) { printf("Bot: %s\n", reply); free(reply); }
     *   4. 打印 "bye!"
     *
     * 注意：fgets 读不到东西（Ctrl+D / EOF）时返回 NULL，记得 break
     */
    char buf[1024];
    printf("BPE Chat (type /quit to exit)\n");
    (void)model;
    (void)tok;
    (void)cache;
    (void)buf;
    printf("(student_chat_loop not implemented yet)\n");
}

/* ============================================================
 * main：跑 verify，不直接进入 REPL
 * ============================================================ */
int main(void) {
    return verify_run_all();
}
