#ifndef STUDENT_H
#define STUDENT_H

#include "bpe_tokenizer.h"
#include "model.h"

/* 把 user_input 编码为 token ID 序列（带 BOS、不带 EOS）。
 * 返回值由 bpe_encode 内部 malloc，调用者 free。 */
int* student_bpe_chat_tokenize(BPETokenizer* tok,
                               const char* user_input,
                               int* out_len);

/* 把 token_ids 解码为可打印字符串。
 * 返回值由 bpe_decode 内部 malloc，调用者 free。 */
char* student_bpe_chat_decode(BPETokenizer* tok,
                              int* token_ids,
                              int len);

/* 单轮对话：把 user_input 喂进模型，采样 max_new_tokens 个 token 并解码。
 * 返回助手回复字符串（malloc，调用者 free）。
 * 失败返回 NULL。 */
char* student_bpe_chat_one_turn(GPTModel* model,
                                BPETokenizer* tok,
                                GPTCache* cache,
                                const char* user_input,
                                int max_new_tokens);

/* 主循环：读 stdin 一行，调用 one_turn，打印回复，/quit 退出。 */
void student_chat_loop(GPTModel* model,
                       BPETokenizer* tok,
                       GPTCache* cache);

#endif /* STUDENT_H */
