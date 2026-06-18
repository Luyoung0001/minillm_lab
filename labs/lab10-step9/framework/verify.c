/*
 * verify.c - Lab10 自动验证
 *
 * 5 个测试, 全部仅依赖 student.c 暴露的 student_http_parse_request_line
 * 和 student_http_build_response. 不直接调 framework 内部 API.
 */
#include "verify.h"
#include "student.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- helper: 跑一个子测试 ---- */
#define RUN_TEST(fn) do { \
    n_tests++; \
    if (fn()) { n_pass++; printf("  [PASS]\n"); } \
    else      {            printf("  [FAIL]\n"); } \
} while (0)

static int n_tests = 0;
static int n_pass  = 0;

/* ---- [TEST 1] GET 请求行 ---- */
static int test_parse_get(void) {
    char m[16] = {0}, p[64] = {0}, v[16] = {0};
    int rc = student_http_parse_request_line(
        "GET /health HTTP/1.1", m, 16, p, 64, v, 16);
    printf("[TEST 1] request line parse: GET /health HTTP/1.1 ... ");
    if (rc != 0) return 0;
    if (strcmp(m, "GET") != 0)         return 0;
    if (strcmp(p, "/health") != 0)     return 0;
    if (strcmp(v, "HTTP/1.1") != 0)    return 0;
    return 1;
}

/* ---- [TEST 2] POST 请求行 ---- */
static int test_parse_post(void) {
    char m[16] = {0}, p[64] = {0}, v[16] = {0};
    int rc = student_http_parse_request_line(
        "POST /v1/chat/completions HTTP/1.1", m, 16, p, 64, v, 16);
    printf("[TEST 2] request line parse: POST /v1/chat/completions HTTP/1.1 ... ");
    if (rc != 0) return 0;
    if (strcmp(m, "POST") != 0)                       return 0;
    if (strcmp(p, "/v1/chat/completions") != 0)       return 0;
    if (strcmp(v, "HTTP/1.1") != 0)                    return 0;
    return 1;
}

/* ---- [TEST 3] 非法请求行 (没有第二个空格) ---- */
static int test_parse_invalid(void) {
    char m[16] = {0}, p[64] = {0}, v[16] = {0};
    int rc = student_http_parse_request_line(
        "GETMALFORMED", m, 16, p, 64, v, 16);
    printf("[TEST 3] request line parse: invalid (no second space) ... ");
    /* 期望: 返回 -1 (失败) */
    return (rc == -1) ? 1 : 0;
}

/* ---- [TEST 4] 拼 200 OK + JSON body ---- */
static int test_build_200(void) {
    const char* body = "{\"status\":\"ok\"}";
    char* resp = student_http_build_response(200, "OK", "application/json", body);
    printf("[TEST 4] response build: 200 OK with JSON body ... ");
    if (resp == NULL) return 0;

    int ok = 1;
    /* 状态行 */
    if (strncmp(resp, "HTTP/1.1 200 OK\r\n", 17) != 0) ok = 0;
    /* Content-Type 头 (大小写不敏感) */
    if (!strstr(resp, "Content-Type:")) ok = 0;
    if (!strstr(resp, "application/json")) ok = 0;
    /* Content-Length: 15 (strlen("{\"status\":\"ok\"}")) */
    if (!strstr(resp, "Content-Length: 15")) ok = 0;
    /* headers 与 body 之间是 \r\n\r\n */
    if (!strstr(resp, "\r\n\r\n")) ok = 0;
    /* body 在最后 */
    if (strcmp(resp + strlen(resp) - strlen(body), body) != 0) ok = 0;

    free(resp);
    return ok;
}

/* ---- [TEST 5] 拼 404 Not Found ---- */
static int test_build_404(void) {
    char* resp = student_http_build_response(404, "Not Found",
                                             "application/json",
                                             "{\"error\":\"missing\"}");
    printf("[TEST 5] response build: 404 Not Found ... ");
    if (resp == NULL) return 0;

    int ok = 1;
    if (strncmp(resp, "HTTP/1.1 404 Not Found\r\n", 24) != 0) ok = 0;
    if (!strstr(resp, "Content-Length: 22")) ok = 0;
    if (!strstr(resp, "{\"error\":\"missing\"}")) ok = 0;
    /* 用 \r\n 行结束 (不是 \n) */
    if (!strstr(resp, "\r\n")) ok = 0;

    free(resp);
    return ok;
}

int verify_run_all(void) {
    n_tests = 0;
    n_pass  = 0;

    printf("Lab10 verify — HTTP request line & response build\n");
    printf("---------------------------------------------------\n");

    RUN_TEST(test_parse_get);
    RUN_TEST(test_parse_post);
    RUN_TEST(test_parse_invalid);
    RUN_TEST(test_build_200);
    RUN_TEST(test_build_404);

    printf("---------------------------------------------------\n");
    printf("Result: %d / %d passed\n", n_pass, n_tests);
    if (n_pass != n_tests) {
        printf("Some tests are still failing.\n");
        return 1;
    }
    printf("All tests done.\n");
    return 0;
}
