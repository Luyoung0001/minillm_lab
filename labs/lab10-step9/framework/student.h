#ifndef LAB10_STUDENT_H
#define LAB10_STUDENT_H

/*
 * 解析 HTTP 请求行 ("METHOD PATH HTTP/1.1")，把三段写到调用方提供的 buffer。
 * @param line       输入行 (保证以 \0 结尾, 不含 \r\n)
 * @param method     输出 buffer (method_cap >= 8)
 * @param method_cap method buffer 容量 (含结尾 \0)
 * @param path       输出 buffer (path_cap >= 16)
 * @param path_cap   path buffer 容量 (含结尾 \0)
 * @param version    输出 buffer (version_cap >= 8)
 * @param version_cap version buffer 容量 (含结尾 \0)
 * @return 0 成功, -1 解析失败 (空格数不对 / 某段超长)
 */
int student_http_parse_request_line(const char* line,
                                    char* method,  int method_cap,
                                    char* path,    int path_cap,
                                    char* version, int version_cap);

/*
 * 拼接完整 HTTP 响应帧 (状态行 + 必要 headers + 空行 + body)。
 * @param status_code   200 / 400 / 404 / 500
 * @param status_text   "OK" / "Bad Request" / "Not Found"
 * @param content_type  "application/json" 等
 * @param body          响应体字符串 (可 NULL, 此时 Content-Length: 0)
 * @return 堆上的字符串 (调用方 free), 失败返回 NULL
 */
char* student_http_build_response(int status_code, const char* status_text,
                                  const char* content_type, const char* body);

#endif
