// Minimal HTTP(S) client (WinHTTP on Windows) and HTML helpers for the web tools.
#pragma once
#include <string>
#include <vector>

struct HttpResponse {
    int status = 0;
    std::string body;          // decoded to UTF-8 when the charset is known
    std::string content_type;  // lower case, without parameters
    std::string final_url;     // after redirects
};

// GET (body empty) or POST (form body). Throws on network errors; follows redirects.
HttpResponse http_request(const std::string& url, const std::string& form_body = "", int timeout_s = 20,
                          size_t max_bytes = 8u << 20, const std::string& extra_headers = "");  // "Name: value\r\n"...

std::string url_encode(const std::string& s);
std::string url_decode(const std::string& s);
std::string html_decode_entities(const std::string& s);
std::string strip_tags(const std::string& html);  // inline fragment -> text
std::string html_to_text(const std::string& html, std::string* title = nullptr);  // page -> readable text
