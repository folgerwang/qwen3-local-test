// Byte-level BPE tokenizer (Qwen tokenizer.json), with NFC normalization and the Qwen pre-tokenizer
// regex implemented by hand.
#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

class Tokenizer {
public:
    void load(const std::string& model_dir);
    // From GGUF metadata (tokenizer.ggml.*): `types` marks the non-normal (!= 1) tokens matched literally.
    void load_gguf(const std::vector<std::string>& tokens, const std::vector<uint64_t>& types,
                   const std::vector<std::string>& merges);
    // parse_special: recognize added tokens like <|im_start|> in the text.
    std::vector<int> encode(const std::string& text, bool parse_special = true) const;
    // Raw bytes of a token (may be a partial UTF-8 sequence).
    std::string decode(int id) const;
    int token_id(const std::string& s) const;
    size_t vocab_size() const { return id_to_token_.size(); }

    // Exposed for tests.
    static std::string nfc(const std::string& utf8);
    static std::vector<std::string> pretokenize(const std::string& utf8);

private:
    void bpe(const std::string& piece, std::vector<int>& out) const;
    std::vector<uint32_t> init_byte_map();  // fills byte_to_unicode_, returns unicode -> byte (0xFFFF = none)
    void add_merge(const std::string& a, const std::string& b, int rank);

    std::unordered_map<std::string, int> vocab_;       // byte-level unicode string -> id
    std::vector<std::string> id_to_token_;            // id -> raw bytes
    std::unordered_map<uint64_t, std::pair<int, int>> merges_;  // (left id, right id) -> (rank, merged id)
    std::vector<std::pair<std::string, int>> added_;   // literal added tokens, longest first
    std::string byte_to_unicode_[256];
    mutable std::unordered_map<std::string, std::vector<int>> cache_;
};
