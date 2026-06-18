/*
 * Lab02 verify - 自动验证程序
 *
 * 4 个 [TEST] 必须全部 [PASS].
 *
 * 关键约束: 本文件只调用 student.c 暴露的 3 个函数 + 公开的
 * framework API (tokenizer_create / tokenizer_free /
 * tokenizer_vocab_size / tokenizer_is_special).
 *
 * 不读 Tokenizer 结构体内部字段. 不使用全局状态.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "student.h"
#include "tokenizer.h"

/* ---------- 测试工具宏 ---------- */
#define TEST_BEGIN(n)                                                       \
    do {                                                                     \
        printf("[TEST %d] ", (n));                                           \
        fflush(stdout);                                                      \
    } while (0)
#define TEST_PASS()                                                          \
    do {                                                                     \
        printf(" ... [PASS]\n");                                             \
    } while (0)
#define TEST_FAIL(fmt, ...)                                                  \
    do {                                                                     \
        printf(" ... [FAIL] " fmt "\n", ##__VA_ARGS__);                      \
        g_failures++;                                                        \
    } while (0)

static int g_failures = 0;     /* 失败计数 (模块内静态, 不是 framework 状态) */

/* ---------- TEST 1: 词汇表大小 ---------- */
static void test_vocab_size(void) {
    TEST_BEGIN(1);
    Tokenizer* tok = tokenizer_create();
    if (tok == NULL) { TEST_FAIL("tokenizer_create returned NULL"); return; }

    int vs = tokenizer_vocab_size(tok);
    const int expected = NUM_SPECIAL_TOKENS + 256;   /* = 260 */
    if (vs != expected) {
        TEST_FAIL("vocab_size=%d, expected %d", vs, expected);
    } else {
        TEST_PASS();
    }
    tokenizer_free(tok);
}

/* ---------- TEST 2: 字符编码 (H=76, i=109) ---------- */
static void test_encode_char(void) {
    TEST_BEGIN(2);
    Tokenizer* tok = tokenizer_create();
    if (tok == NULL) { TEST_FAIL("tokenizer_create returned NULL"); return; }

    int id_H = student_encode_char(tok, 'H');   /* expect 72 + 4 = 76 */
    int id_i = student_encode_char(tok, 'i');   /* expect 105 + 4 = 109 */
    if (id_H != 76 || id_i != 109) {
        TEST_FAIL("encode('H')=%d (want 76), encode('i')=%d (want 109)",
                  id_H, id_i);
    } else {
        TEST_PASS();
    }
    tokenizer_free(tok);
}

/* ---------- TEST 3: ID 解码 (76->H, 109->i) ---------- */
static void test_decode_id(void) {
    TEST_BEGIN(3);
    Tokenizer* tok = tokenizer_create();
    if (tok == NULL) { TEST_FAIL("tokenizer_create returned NULL"); return; }

    char c_H = student_decode_id(tok, 76);   /* expect 'H' */
    char c_i = student_decode_id(tok, 109);  /* expect 'i' */
    if (c_H != 'H' || c_i != 'i') {
        TEST_FAIL("decode(76)='%c' (want 'H'), decode(109)='%c' (want 'i')",
                  c_H, c_i);
    } else {
        TEST_PASS();
    }
    tokenizer_free(tok);
}

/* ---------- TEST 4: 往返一致 ("Hello") ---------- */
static void test_roundtrip_hello(void) {
    TEST_BEGIN(4);
    Tokenizer* tok = tokenizer_create();
    if (tok == NULL) { TEST_FAIL("tokenizer_create returned NULL"); return; }

    char buf[32];
    int n = student_roundtrip(tok, "Hello", buf, sizeof(buf));
    if (n != 5) {
        TEST_FAIL("roundtrip returned len=%d (want 5)", n);
    } else if (strcmp(buf, "Hello") != 0) {
        TEST_FAIL("roundtrip got [%s] (want [Hello])", buf);
    } else {
        TEST_PASS();
    }
    tokenizer_free(tok);
}

/* ---------- 入口 ---------- */
int verify_run_all(void) {
    g_failures = 0;
    printf("==========================================\n");
    printf("       Lab02 - 字符级 Tokenizer 验证      \n");
    printf("==========================================\n");

    test_vocab_size();
    test_encode_char();
    test_decode_id();
    test_roundtrip_hello();

    printf("==========================================\n");
    if (g_failures == 0) {
        printf("All tests passed!\n");
        return 0;
    } else {
        printf("%d test(s) FAILED.\n", g_failures);
        return 1;
    }
}
