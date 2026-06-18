#include "student.h"
#include "verify.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

/*
 * ========== TODO 1: 解析 HTTP 请求行 ==========
 *
 * 输入示例: "GET /health HTTP/1.1"
 * 目标: method="GET"  path="/health"  version="HTTP/1.1"
 *
 * 提示:
 *   - 找两个 ' ' (strchr), 把它们改成 '\0', 再 strncpy 到三个 buffer
 *   - 任何段 >= 对应 _cap 时返回 -1 (不要无声截断)
 *   - 找不到第二个空格时返回 -1
 */
int student_http_parse_request_line(const char* line,
                                    char* method,  int method_cap,
                                    char* path,    int path_cap,
                                    char* version, int version_cap) {
    // TODO(student): 解析 "METHOD PATH HTTP/1.1" 到三个 buffer
    //                1) 拷贝 line 到一个可改写的本地 buffer
    //                2) 找第一个空格, 切出 method
    //                3) 找第二个空格, 切出 path 和 version
    //                4) 段长检查: 超过 _cap 返回 -1
    //                5) strncpy 三段, 返回 0
    (void)line; (void)method; (void)method_cap;
    (void)path;  (void)path_cap; (void)version; (void)version_cap;
    return -1;
}

/*
 * ========== TODO 2: 拼装 HTTP 响应帧 ==========
 *
 * 输出示例 (status=200, status_text="OK", content_type="application/json", body="{\"status\":\"ok\"}"):
 *
 *   HTTP/1.1 200 OK\r\n
 *   Content-Type: application/json\r\n
 *   Content-Length: 15\r\n
 *   Connection: close\r\n
 *   \r\n
 *   {"status":"ok"}
 *
 * 提示:
 *   - 行结束符必须 \r\n (HTTP 规定)
 *   - headers 与 body 之间是 \r\n\r\n (一个空行)
 *   - body == NULL 时, Content-Length: 0, 仍要写空行
 *   - 用 snprintf + 最终 strcat 或一次 snprintf 到动态 buffer
 *   - 返回值必须 malloc, 失败返回 NULL
 */
char* student_http_build_response(int status_code, const char* status_text,
                                  const char* content_type, const char* body) {
    // TODO(student): 拼出 "HTTP/1.1 %d %s\r\nContent-Type: ...\r\n..." 字符串
    //                至少包含 Content-Type / Content-Length / Connection 三个头
    //                返回 malloc 出来的字符串
    (void)status_code; (void)status_text; (void)content_type; (void)body;
    char* empty = (char*)malloc(1);
    if (empty) empty[0] = '\0';
    return empty;
}

/* ====== 入口: 学生不需要改 main ====== */
int main(void) {
    return verify_run_all();
}
