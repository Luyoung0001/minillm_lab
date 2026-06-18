/*
 * Lab02 - 字符级 Tokenizer
 *
 * 学员要做的全部代码都在本文件里. 框架 (../../framework/...) 走 Makefile 链接.
 *
 * 你要实现的 3 个函数 (见 student.h):
 *   1) student_encode_char
 *   2) student_decode_id
 *   3) student_roundtrip
 *
 * 规则提醒:
 *   - ID 0~3   : 特殊 token (PAD / UNK / BOS / EOS)
 *   - ID 4~259 : ASCII 0~255,  关系: id = ASCII + 4
 *   - 因此 ASCII = id - 4
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "student.h"
#include "verify.h"
#include "tokenizer.h"   /* 共享的 Tokenizer 类型与常量 */

/* ============================================================
 * TODO(student): 实现 student_encode_char
 *
 * 签名: int student_encode_char(Tokenizer* tok, char c);
 *
 * 行为:
 *   - 若 tok == NULL 或 c < 0  -> 返回 TOKEN_UNK
 *   - 否则返回 (unsigned char)c + NUM_SPECIAL_TOKENS
 *
 * 提示:  写 1~3 行即可. NUM_SPECIAL_TOKENS 在 tokenizer.h 里是 4.
 * ============================================================ */
int student_encode_char(Tokenizer* tok, char c) {
    /* TODO(student): replace this stub with your 1~3 line implementation */
    (void)tok;
    (void)c;
    return TOKEN_UNK;   /* 占位: 永远返回 UNK, 让 verify 失败提醒你来改 */
}

/* ============================================================
 * TODO(student): 实现 student_decode_id
 *
 * 签名: char student_decode_id(Tokenizer* tok, int id);
 *
 * 行为:
 *   - 若 id <  NUM_SPECIAL_TOKENS  (即 0~3 特殊 token)   -> 返回 '\0'
 *   - 若 id >= tok->vocab_size                          -> 返回 '\0'
 *   - 否则返回 (char)(id - NUM_SPECIAL_TOKENS)
 *
 * 提示:  写 4~6 行. 不要忘记 NULL 保护.
 * ============================================================ */
char student_decode_id(Tokenizer* tok, int id) {
    /* TODO(student): replace this stub with your 4~6 line implementation */
    (void)tok;
    (void)id;
    return '\0';        /* 占位: 让 verify 失败提醒你来改 */
}

/* ============================================================
 * TODO(student): 实现 student_roundtrip
 *
 * 签名: int student_roundtrip(Tokenizer* tok, const char* text,
 *                              char* out, int out_size);
 *
 * 行为:
 *   1) tokenizer_encode(tok, text, &len, 0, 0)  -> ids
 *   2) 逐 id 调 student_decode_id 还原, 写入临时 tmp 缓冲
 *   3) tmp[len] = '\0'
 *   4) strncpy(out, tmp, out_size - 1);  out[out_size-1] = '\0'
 *   5) free(ids); free(tmp);  return len
 *
 * 边界:
 *   - text == NULL || out == NULL || out_size <= 1  -> return 0, out[0]='\0'
 *
 * 提示:  写 8~12 行. 注意 free 不能漏.
 * ============================================================ */
int student_roundtrip(Tokenizer* tok, const char* text,
                      char* out, int out_size) {
    /* TODO(student): replace this stub with your 8~12 line implementation */
    (void)tok;
    (void)text;
    if (out != NULL && out_size > 0) {
        out[0] = '\0';
    }
    return 0;           /* 占位: 让 verify 失败提醒你来改 */
}

/* ============================================================
 * main - 把控制权交给 verify
 * 学员不要改 main. 想加临时 printf 调试可以临时加, 提交前删掉.
 * ============================================================ */
int main(void) {
    return verify_run_all();
}
