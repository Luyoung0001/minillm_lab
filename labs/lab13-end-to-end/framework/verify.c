/* verify.c - Lab13 自动验证
 *
 * 设计原则：
 *   - 所有输入从函数参数拿，不依赖 student.c 之外的 framework 状态
 *   - 不直接调 framework 的"内部 API"——只走公开 .h
 *   - 通过 student_bpe_chat_tokenize / decode / one_turn 来间接测 BPE
 *   - 通过 stdin pipe 测 chat_loop
 */

#include "verify.h"
#include "student.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bpe_tokenizer.h"   /* bpe_tokenizer_create / free / train / encode / decode / vocab_size */
#include "model.h"           /* model_create / model_init_random / model_free / model_forward / model_cache_* */
#include "tensor.h"          /* tensor_zeros / tensor_free / tensor_get */
#include "config.h"          /* tiny_config */
#include "generate.h"        /* sample_top_k / sample_greedy */

/* ---------- 计数 ---------- */
static int g_pass = 0;
static int g_fail = 0;

static void check(int cond, const char* label) {
    if (cond) {
        printf("[TEST] %-46s [PASS]\n", label);
        g_pass++;
    } else {
        printf("[TEST] %-46s [FAIL]\n", label);
        g_fail++;
    }
}

/* ---------- 共享 helper：构造一个最小的 BPE 词表 + tokenizer ----------
 *
 * 选这一段小语料是因为：
 *   - "hello world" + "hi there" 能在 < 50 次合并内搞定
 *   - vocab_size 落在 ~50 左右，远小于 BPE_MAX_VOCAB_SIZE=8192
 *   - 足以让 bpe_encode / bpe_decode 在测试里跑出非空结果
 */
static BPETokenizer* make_test_tokenizer(void) {
    BPETokenizer* tok = bpe_tokenizer_create();
    if (!tok) return NULL;
    const char* corpus =
        "hello world "
        "hello there "
        "hi world "
        "hi there "
        "good morning "
        "good night "
        "how are you "
        "i am fine thank you";
    if (bpe_train(tok, corpus, 50) != 0) {
        bpe_tokenizer_free(tok);
        return NULL;
    }
    return tok;
}

/* ---------- 共享 helper：构造一个最小可用的 GPTModel ----------
 *
 * tiny_config() = vocab=260, hidden=32, heads=2, layers=2, ffn=128, max_seq=64
 * 随机初始化权重，足够让 model_forward 不段错误跑出 logits。
 */
static GPTModel* make_test_model(void) {
    ModelConfig cfg = tiny_config();
    GPTModel* m = model_create(cfg);
    if (!m) return NULL;
    model_init_random(m, 0.02f);
    return m;
}

/* ---------- TEST 1：BPE 编码非空 ---------- */
static void test_tokenize_nonempty(void) {
    BPETokenizer* tok = make_test_tokenizer();
    int pass = 0;
    if (tok) {
        int len = -1;
        int* ids = student_bpe_chat_tokenize(tok, "hello", &len);
        pass = (ids != NULL) && (len > 0);
        if (ids) free(ids);
        bpe_tokenizer_free(tok);
    }
    check(pass, "bpe_chat_tokenize 编码非空");
}

/* ---------- TEST 2：BPE 解码可逆 ----------
 *
 * 编码 "hi"，再解码回来。字符串不要求完全一致（BPE 词表可能有特殊 token），
 * 但解码后的字符串应当非空，且长度 >= 1。
 */
static void test_decode_roundtrip(void) {
    BPETokenizer* tok = make_test_tokenizer();
    int pass = 0;
    if (tok) {
        int len = 0;
        int* ids = student_bpe_chat_tokenize(tok, "hi", &len);
        if (ids && len > 0) {
            char* text = student_bpe_chat_decode(tok, ids, len);
            pass = (text != NULL) && (strlen(text) > 0);
            if (text) free(text);
        }
        if (ids) free(ids);
        bpe_tokenizer_free(tok);
    }
    check(pass, "bpe_chat_decode 解码可逆");
}

/* ---------- TEST 3：单轮对话 ----------
 *
 * 给一个空生成的模型 + 一个小 BPE，student_bpe_chat_one_turn 应当
 * 不段错误地返回一个非空字符串。模型是随机的——文本内容无所谓。
 */
static void test_one_turn_does_not_crash(void) {
    BPETokenizer* tok = make_test_tokenizer();
    GPTModel* model = make_test_model();
    GPTCache* cache = NULL;
    char* reply = NULL;
    int pass = 0;
    if (tok && model) {
        cache = model_cache_create(model, /*seq_len=*/16);
        if (cache) {
            reply = student_bpe_chat_one_turn(model, tok, cache,
                                              "hello", /*max_new=*/8);
            pass = (reply != NULL) && (strlen(reply) > 0);
        }
    }
    if (reply) free(reply);
    if (cache) model_cache_free(cache);
    if (model) model_free(model);
    if (tok)   bpe_tokenizer_free(tok);
    check(pass, "bpe_chat_one_turn 单轮生成");
}

/* ---------- TEST 4：chat_loop 走完 1 轮 stdin ----------
 *
 * 用 freopen 把 stdin 替换成一段内存字符串，喂入一行 prompt + /quit，
 * 然后恢复原 stdin。student_chat_loop 应当不卡死地退出。
 */
static void test_chat_loop_one_turn(void) {
    BPETokenizer* tok = make_test_tokenizer();
    GPTModel* model = make_test_model();
    GPTCache* cache = NULL;

    /* 备份 stdin，构造一个临时输入文件 */
    FILE* bak_stdin = stdin;
    FILE* tmp = tmpfile();
    int pass = 0;
    if (tok && model && tmp) {
        fputs("hello\n/quit\n", tmp);
        rewind(tmp);
        stdin = tmp;

        cache = model_cache_create(model, /*seq_len=*/16);
        /* 给 chat_loop 6 秒软上限（防止学员写死循环） */
        /* 简单做法：直接调，靠 one_turn 内部的 max_new_tokens=32 限速 */
        student_chat_loop(model, tok, cache);
        pass = 1;   /* 不段错误就算通过 */

        stdin = bak_stdin;
        fclose(tmp);
    }
    if (cache) model_cache_free(cache);
    if (model) model_free(model);
    if (tok)   bpe_tokenizer_free(tok);
    check(pass, "chat_loop 走完 1 轮 stdin");
}

/* ---------- TEST 5：全函数空指针守护 ----------
 *
 * 任何 student_xxx(NULL, ...) 不应段错误。这里只验证能返回（NULL / 不崩）。
 * 不严格检查返回值——守护"是否存在"即可。
 */
static void test_null_guards(void) {
    int pass = 1;
    int dummy_len = 0;

    /* tokenize(NULL, ...) 不应崩 */
    int* r1 = student_bpe_chat_tokenize(NULL, "hi", &dummy_len);
    pass = pass && (r1 == NULL);

    /* decode(NULL, ...) 不应崩 */
    char* r2 = student_bpe_chat_decode(NULL, NULL, 0);
    pass = pass && (r2 == NULL);

    /* one_turn(NULL, ...) 不应崩 */
    char* r3 = student_bpe_chat_one_turn(NULL, NULL, NULL, "hi", 4);
    pass = pass && (r3 == NULL);

    /* chat_loop(NULL, ...) 不应崩——只调一下 */
    student_chat_loop(NULL, NULL, NULL);
    /* 走到这里没段错误就算过 */

    if (r1) free(r1);
    if (r2) free(r2);
    if (r3) free(r3);

    check(pass, "全函数空指针守护");
}

/* ---------- 入口 ---------- */
int verify_run_all(void) {
    g_pass = 0;
    g_fail = 0;

    printf("========================================\n");
    printf("      Lab13 - 端到端 BPE 对话测试\n");
    printf("========================================\n");

    test_tokenize_nonempty();
    test_decode_roundtrip();
    test_one_turn_does_not_crash();
    test_chat_loop_one_turn();
    test_null_guards();

    printf("========================================\n");
    printf("测试结果: %d 通过, %d 失败\n", g_pass, g_fail);
    printf("========================================\n");

    return (g_fail == 0) ? 0 : 1;
}
