// Tools the model can call from the chat: files in the working folder, PowerShell commands, and the web.
// Reading, listing, searching and web access run freely; anything that changes files or runs a command
// goes through an approval callback first.
#pragma once
#include <filesystem>
#include <functional>
#include <string>
#include <utility>
#include <vector>

struct ToolCall {
    std::string name;  // empty when the block could not be parsed (args[0] then holds the problem)
    std::vector<std::pair<std::string, std::string>> args;
    const std::string* arg(const std::string& key) const;
};

// Tool calls in an assistant message, in the Qwen format:
// <tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n</function>\n</tool_call>
std::vector<ToolCall> parse_tool_calls(const std::string& text);

struct ToolResult {
    std::string text;     // content of the <tool_response>
    std::string image;    // image file shown to the model with the response ("" = none)
    std::string summary;  // one line for the user
    std::string detail;   // output worth showing the user (command output, diff), may be long
    bool ok = true;
};

class Tools {
public:
    // Asked before an action that changes something. Returns true to go ahead; a denial may come with the
    // user's reason in `note`, which is passed on to the model.
    using Approver = std::function<bool(const std::string& tool, const std::string& title, const std::string& preview,
                                        std::string& note)>;

    Tools(std::filesystem::path root, bool images);
    std::string system_prompt() const;  // the "# Tools" system block of the chat template
    std::string context_note() const;   // working folder, date and shell, for the system message
    std::string describe(const ToolCall& c) const;  // short one-line form for the transcript
    ToolResult run(const ToolCall& c, const Approver& approve);
    void set_cancel(std::function<bool()> f) { cancelled_ = std::move(f); }
    void set_images(bool on) { images_ = on; }
    const std::filesystem::path& root() const { return root_; }

    // Undo: file changes made by write_file / edit_file / delete_path are backed up first and grouped per
    // turn (begin_turn .. end_turn). undo() reverts the latest group and describes what it did, one line per
    // file ("" when there is nothing to undo). Commands are not undoable.
    void begin_turn() { current_.clear(); }
    void end_turn();
    std::vector<std::string> undo();
    size_t undo_depth() const { return undo_.size(); }

private:
    struct Backup {
        std::filesystem::path path;
        bool existed = false, dir = false, kept = true;   // kept = false: too large to back up
        std::string bytes;                                  // file content
        std::vector<std::pair<std::string, std::string>> tree;  // folder: relative path -> content ("/" suffix = folder)
    };
    void backup(const std::filesystem::path& p);
    std::vector<Backup> current_;
    std::vector<std::vector<Backup>> undo_;

    std::filesystem::path resolve(const std::string& p) const;  // throws outside the working folder
    std::string rel(const std::filesystem::path& p) const;

    ToolResult list_dir(const ToolCall& c);
    ToolResult read_file(const ToolCall& c);
    ToolResult search_text(const ToolCall& c);
    ToolResult write_file(const ToolCall& c, const Approver& approve);
    ToolResult edit_file(const ToolCall& c, const Approver& approve);
    ToolResult delete_path(const ToolCall& c, const Approver& approve);
    ToolResult run_command(const ToolCall& c, const Approver& approve);
    ToolResult web_search(const ToolCall& c);
    ToolResult fetch_url(const ToolCall& c);

    std::filesystem::path root_;
    bool images_;
    std::string shell_;  // pwsh.exe or powershell.exe
    std::function<bool()> cancelled_;
};
