/*
 * Lab09 — verify.c
 *
 * 自动验证程序。学员**不要改**这个文件。
 * 每个测试只能调 student.c 暴露的 student_format_prompt / student_extract_response
 * （以及 framework 里字符串工具 trim_string / starts_with / ends_with），
 * 不能调 framework/chat.c 的内部 API（如 conversation_create / format_prompt_with_input），
 * 否则测试会"绕过"学员的代码、失去验证意义。
 */

#include "verify.h"
#include "student.h"

#include "chat.h"   /* 4 个标签宏 + trim_string / starts_with / ends_with */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond, label) do {                                          \
    if (cond) {                                                          \
        printf("[TEST %d] %s ... [PASS]\n", g_pass + g_fail + 1, label); \
        g_pass++;                                                        \
    } else {                                                             \
        printf("[TEST %d] %s ... [FAIL]\n", g_pass + g_fail + 1, label); \
        g_fail++;                                                        \
    }                                                                    \
} while (0)

/* 统计子串出现次数（不重叠） */
static int count_occurrences(const char* hay, const char* needle) {
    if (hay == NULL || needle == NULL || needle[0] == '\0') return 0;
    int n = 0;
    const char* p = hay;
    size_t nlen = strlen(needle);
    while ((p = strstr(p, needle)) != NULL) {
        n++;
        p += nlen;
    }
    return n;
}

/* ---------- Prompt 格式化 ---------- */

static void test_format_prompt_single(void) {
    printf("\n=== Prompt 格式化 ===\n");

    const char* roles[]    = { "user" };
    const char* contents[] = { "hi" };
    char* p = student_format_prompt(roles, contents, 1);
    CHECK(p != NULL && count_occurrences(p, CHAT_USER_TAG) == 1
                   && count_occurrences(p, CHAT_ASSISTANT_TAG) == 1
                   && strstr(p, "hi") != NULL
                   && ends_with(p, "<|assistant|>\n"),
          "1 turn 1 assistant tag");
    free(p);
}

static void test_format_prompt_multi(void) {
    const char* roles[]    = { "user", "assistant", "user" };
    const char* contents[] = { "hi", "hello", "how are you" };
    char* p = student_format_prompt(roles, contents, 3);
    CHECK(p != NULL && count_occurrences(p, CHAT_USER_TAG) == 2
                   && count_occurrences(p, CHAT_ASSISTANT_TAG) == 2
                   && strstr(p, "hi") != NULL
                   && strstr(p, "hello") != NULL
                   && strstr(p, "how are you") != NULL,
          "3 turns 4 user/assistant");
    free(p);
}

static void test_format_prompt_trailing(void) {
    const char* roles[]    = { "user" };
    const char* contents[] = { "x" };
    char* p = student_format_prompt(roles, contents, 1);
    /* 末尾必须是 <|assistant|>，让模型知道接龙点 */
    CHECK(p != NULL && ends_with(p, "<|assistant|>\n"),
          "trailing assistant present");
    free(p);
}

/* ---------- 响应提取 ---------- */

static void test_extract_no_tag(void) {
    printf("\n=== 响应提取 ===\n");

    /* 完全没有 <|assistant|>，函数应当把整段 trim 后返回 */
    const char* out = "   hello world   ";
    char* r = student_extract_response(out);
    CHECK(r != NULL && strcmp(r, "hello world") == 0,
          "no tag returns trimmed");
    free(r);
}

static void test_extract_last_tag(void) {
    /* 两个 <|assistant|>：取"最后一个" */
    const char* out =
        "<|user|>\nA\n<|assistant|>\nWRONG\n<|user|>\nB\n<|assistant|>\nRIGHT\n";
    char* r = student_extract_response(out);
    CHECK(r != NULL && strcmp(r, "RIGHT") == 0,
          "last-of-two tags wins");
    free(r);
}

static void test_extract_user_boundary(void) {
    /* 末尾还有 <|user|>，截到那里停 */
    const char* out = "<|assistant|>\nactual reply\n<|user|>\nnext q\n";
    char* r = student_extract_response(out);
    CHECK(r != NULL && strcmp(r, "actual reply") == 0,
          "stops at <|user|> boundary");
    free(r);
}

int verify_run_all(void) {
    test_format_prompt_single();
    test_format_prompt_multi();
    test_format_prompt_trailing();

    test_extract_no_tag();
    test_extract_last_tag();
    test_extract_user_boundary();

    printf("\n========================================\n");
    printf("测试结果: %d 通过, %d 失败\n", g_pass, g_fail);
    printf("========================================\n");

    return (g_fail == 0) ? 0 : 1;
}
