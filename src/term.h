// Terminal UI for the interactive chat: ANSI styling, Ctrl+C handling, multi-line input, streaming
// markdown rendering and a "thinking" spinner. Everything degrades to plain text when stdout is not a
// terminal (or NO_COLOR is set), so piped output stays exactly what the model produced.
#pragma once
#include <chrono>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace term {

// UTF-8 console output and ANSI colors when stdout is a terminal.
void init();
bool color();

// While enabled, Ctrl+C sets a flag (checked with interrupted()) instead of terminating the process.
void catch_ctrl_c(bool on);
bool interrupted();
void clear_interrupt();

// ANSI SGR escape for `code` ("1", "2;3", "36", ...), or "" when colors are off.
std::string sgr(const char* code);

// Color themes: SGR codes per role ("" = no styling for that role). /theme switches, --theme sets one.
struct Theme {
    const char* name;
    const char* desc;
    const char *accent, *dim, *ok, *warn, *err, *heading, *code, *fence;
};
const std::vector<Theme>& themes();
const Theme& theme();
bool set_theme(const std::string& name);  // false if there is no such theme

// Terminal geometry and text measuring (display columns; escape sequences count as 0, CJK and emoji as 2).
int columns();
int width(const std::string& utf8);
std::string fit(const std::string& plain, int cols);  // cut to `cols` columns with an ellipsis (no escapes inside)

// True when menus can be shown: output and input are both a terminal.
bool interactive();

// Arrow-key menu (up/down or j/k, Enter, a digit to pick directly, Esc to cancel). Returns the chosen index
// or -1; the menu is erased afterwards. `current` gets a check mark. `on_move` runs when the highlighted row
// changes (live preview); `footer` returns extra lines (each ending in a newline) shown under the list.
struct Choice { std::string label, desc; };
int select(const std::string& title, const std::vector<Choice>& items, int current,
           const std::function<void(int)>& on_move = {}, const std::function<std::string(int)>& footer = {});

enum class Input { Line, Cancel, Eof };
// Prints `prompt` and reads one message. Pasted multi-line text arrives as one message, a line ending in
// '\' continues on the next line, and a line holding only """ opens / closes a multi-line block.
Input read_message(const std::string& prompt, std::string& out);

// Writes bytes to stdout as they come, holding back incomplete UTF-8 sequences.
class Utf8Out {
public:
    void put(const std::string& bytes);
    void flush();

private:
    std::string pending_;
};

// Streaming markdown renderer: fenced code blocks, headings, bullets, **bold** and `code`. Disabled, it
// passes the text through unchanged. `dim` renders everything faint (used for the reasoning).
class Markdown {
public:
    explicit Markdown(bool enabled, bool dim = false) : on_(enabled), dim_(dim) {}
    void put(const std::string& text);
    void finish();  // flush held-back characters and reset the styles
    // Text put in front of the first line (a bullet, with its own styling) and of every later line (indentation).
    void prefix(std::string first, std::string rest) { pfx_first_ = std::move(first); pfx_rest_ = std::move(rest); }

private:
    std::string indent(const std::string& o);
    void ch(char c, std::string& o);
    void inline_ch(char c, std::string& o);
    void line_start(std::string& o);
    std::string style() const;

    bool on_, dim_;
    bool started_ = false, bol_ = true, fence_ = false, fence_line_ = false;
    bool code_ = false, bold_ = false, star_ = false, heading_ = false;
    std::string hold_;  // start of the current line, held until its kind (fence, heading, bullet) is known
    std::string pfx_first_, pfx_rest_, last_esc_, cur_esc_;
    bool need_pfx_ = true, first_line_ = true, in_esc_ = false;
    Utf8Out out_;
};

// One-line "thinking" indicator, redrawn in place.
class Spinner {
public:
    explicit Spinner(bool enabled, std::string label = "thinking")
        : on_(enabled), label_(std::move(label)), t0_(std::chrono::steady_clock::now()) {}
    void update(int tokens);
    void clear();
    double seconds() const;

private:
    bool on_, drawn_ = false;
    std::string label_;
    int frame_ = 0;
    std::chrono::steady_clock::time_point t0_, last_{};
};

}  // namespace term
