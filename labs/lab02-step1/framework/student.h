#ifndef STUDENT_H
#define STUDENT_H

#include "tokenizer.h"

/*
 * student_encode_char - 把单个字符编码成 token ID
 *
 * 规则: id = (unsigned char)c + NUM_SPECIAL_TOKENS
 * 例:   'A' (ASCII 65) -> 69
 * 异常: tok 为 NULL 或 c 为负数时返回 TOKEN_UNK
 */
int  student_encode_char(Tokenizer* tok, char c);

/*
 * student_decode_id - 把单个 token ID 解码回字符
 *
 * 例:   69 -> 'A'
 * 异常: id < NUM_SPECIAL_TOKENS 或 id >= vocab_size 时返回 '\0'
 *       (特殊 token 由调用方另行处理)
 */
char student_decode_id(Tokenizer* tok, int id);

/*
 * student_roundtrip - 把 text 编码再解码, 结果写入 out
 *
 * @param tok       tokenizer 句柄
 * @param text      输入字符串
 * @param out       输出缓冲
 * @param out_size  out 容量 (字节)
 * @return 写入 out 的字符数 (不含 '\0'); 失败返回 0
 *
 * 内部: tokenizer_encode (不加 BOS/EOS) -> student_decode_id -> strncpy
 */
int  student_roundtrip(Tokenizer* tok, const char* text,
                       char* out, int out_size);

#endif // STUDENT_H
