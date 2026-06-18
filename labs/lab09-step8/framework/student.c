/*
 * Lab09 — 多轮对话
 *
 * 你只改这个文件。framework/ 下的 .c 共享代码是完整实现，不要动。
 * 重点：所有 framework 代码走 Makefile 链接，不要 include 任何 ../framework/*.c。
 */

#include "student.h"
#include "verify.h"

#include "chat.h"   /* CHAT_USER_TAG / CHAT_ASSISTANT_TAG / ... */

#include <stdlib.h>
#include <string.h>

static char* dup_cstr(const char* s) {
    size_t len = strlen(s) + 1;
    char* out = (char*)malloc(len);
    if (out != NULL) memcpy(out, s, len);
    return out;
}

/* ------------------------------------------------------------------ */
/* 任务 1.5.2: prompt 格式化                                          */
/* ------------------------------------------------------------------ */
/* 把 roles[i] / contents[i] 数组按顺序拼成一段带标签的 prompt。       */
/* 每条消息占 2 行：标签行 + 内容行。                                 */
/* 拼完所有消息后，再追加一行 "<|assistant|>\n"（让模型知道该接龙了）。*/
/* count == 0 时返回 "<|assistant|>\n"。                             */
/* 返回串调用方 free。                                                */
char* student_format_prompt(const char** roles, const char** contents, int count) {
    (void)roles;
    (void)contents;
    (void)count;

    // TODO(student): 用 strdup / malloc+strcat 拼 prompt。5~15 行。
    // 提示：
    //   - 根据 role 选标签：strcmp(role, "user")==0 用 CHAT_USER_TAG
    //   - 每段格式： "<tag>\n<content>\n"
    //   - 末尾追加： "<|assistant|>\n"
    return dup_cstr("");
}

/* ------------------------------------------------------------------ */
/* 任务 1.5.3: 响应提取                                               */
/* ------------------------------------------------------------------ */
/* 在 output 里找"最右一个" <|assistant|>，从它后面开始截，            */
/* 到下一个 <|user|>/<|assistant|>/<|system|>/<|end|> 之前停下；      */
/* 然后用 trim_string 去首尾空白。                                    */
/* 返回串调用方 free。                                                */
char* student_extract_response(const char* output) {
    (void)output;

    // TODO(student): 找最后一个 <|assistant|>，截取尾段，trim。
    // 5~15 行。提示：循环 ptr = strstr(ptr + 1, "<|assistant|>") 一直扫到 NULL。
    return dup_cstr("");
}

/* ------------------------------------------------------------------ */
/* main — 不要改这里                                                 */
/* ------------------------------------------------------------------ */
int main(void) {
    return verify_run_all();
}
