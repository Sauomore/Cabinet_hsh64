//! 对拍测试：Python 与 Rust 的 feat 映射表必须一致
//!
//! # 为什么需要这个测试
//!
//! 中英双语的 `(词性标签 → feat)` 映射表存在于两处：
//! - `scripts/bilingual_pos.py` —— 建码本时用（Python）
//! - `src/pos_map.rs`          —— 查询码本时用（Rust）
//!
//! 两边不一致的后果是：**Python 建出来的码本，Rust 查出来是错的**。
//! 这类 bug 很难发现 —— 单看任何一边都正常，只有端到端才暴露。
//!
//! 所以 Python 侧导出 `tests/data/feat_table.json`，本测试逐项断言。
//!
//! # 数据流
//!
//! ```text
//! scripts/bilingual_pos.py
//!         │  scripts/46_export_feat_table.py
//!         ▼
//! tests/data/feat_table.json
//!         │  本测试读取（不依赖 serde_json，用极简解析器）
//!         ▼
//! src/pos_map.rs  逐项断言
//! ```
//!
//! # 维护方式
//!
//! 改动任一侧后运行：
//! ```text
//! python scripts/46_export_feat_table.py
//! cargo test --test feat_parity
//! ```

use hsh64::pos_map::{
    pos_to_feat_en, pos_to_feat_zh, FeatureCode,
};
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

/// 极简 JSON 解析：只支持 "字符串键": 数字 的扁平对象。
///
/// 刻意不引入 serde_json —— 本 crate 是已发布的库，为了一个测试增加
/// 依赖不划算；而且只用标准库能让 `cargo test` 在离线环境也跑得通。
///
/// ⚠️ 必须做**引号感知**的切分：英文标点标签里有 `,` `""` `''` 这些
/// 含逗号或引号的键。第一版按 `,` 直接 split，在 `"''"` 上就散架了
/// （对拍测试立刻抓到了这个 bug —— 这正是写它的意义）。
///
/// 返回 (键, 值) 列表。遇到不认识的语法直接 panic，避免静默解析错误。
fn parse_string_int_map(json: &str, key: &str) -> BTreeMap<String, u8> {
    // ---- 定位 "key": { ... }（花括号配对，容忍嵌套字符串里的括号）----
    let needle = format!("\"{}\"", key);
    let start = json
        .find(&needle)
        .unwrap_or_else(|| panic!("JSON 里找不到键 {}", key));
    let brace = json[start..]
        .find('{')
        .unwrap_or_else(|| panic!("键 {} 后面没有对象", key))
        + start;

    let bytes = json.as_bytes();
    let mut depth = 0usize;
    let mut end = None;
    let mut in_str = false;
    let mut esc = false;
    for (i, &b) in bytes.iter().enumerate().skip(brace) {
        if esc {
            esc = false;
            continue;
        }
        match b {
            b'\\' if in_str => esc = true,
            b'"' => in_str = !in_str,
            b'{' if !in_str => depth += 1,
            b'}' if !in_str => {
                depth -= 1;
                if depth == 0 {
                    end = Some(i);
                    break;
                }
            }
            _ => {}
        }
    }
    let end = end.unwrap_or_else(|| panic!("键 {} 的对象没有闭合", key));
    let body = &json[brace + 1..end];

    // ---- 逐条目解析："key" : value ----
    let mut out = BTreeMap::new();
    let body_bytes = body.as_bytes();
    let mut i = 0usize;
    while i < body_bytes.len() {
        // 跳过空白与逗号
        while i < body_bytes.len()
            && (body_bytes[i].is_ascii_whitespace() || body_bytes[i] == b',')
        {
            i += 1;
        }
        if i >= body_bytes.len() {
            break;
        }
        // 期望一个字符串键
        if body_bytes[i] != b'"' {
            panic!("条目起始不是引号（偏移 {}）", i);
        }
        i += 1;
        let kstart = i;
        let mut kbuf = String::new();
        while i < body_bytes.len() {
            match body_bytes[i] {
                b'\\' => {
                    // 处理转义（本表只会出现 \" 和 \\）
                    if i + 1 < body_bytes.len() {
                        kbuf.push_str(&body[kstart..i]);
                        let esc_c = body_bytes[i + 1] as char;
                        kbuf.push(match esc_c {
                            'n' => '\n',
                            't' => '\t',
                            other => other,
                        });
                        i += 2;
                        // 继续累积：用 kbuf 记录，后续片段再 push
                        let mut j = i;
                        while j < body_bytes.len() && body_bytes[j] != b'"' {
                            if body_bytes[j] == b'\\' {
                                j += 2;
                            } else {
                                j += 1;
                            }
                        }
                        kbuf.push_str(&body[i..j]);
                        i = j;
                    } else {
                        panic!("键末尾反斜杠未闭合");
                    }
                }
                b'"' => break,
                _ => i += 1,
            }
        }
        if kbuf.is_empty() {
            kbuf = body[kstart..i].to_string();
        }
        if i >= body_bytes.len() {
            panic!("键未闭合: {:?}", kbuf);
        }
        i += 1; // 跳过结束引号

        // 跳过空白与冒号
        while i < body_bytes.len()
            && (body_bytes[i].is_ascii_whitespace() || body_bytes[i] == b':')
        {
            i += 1;
        }
        // 读数字
        let vstart = i;
        while i < body_bytes.len()
            && (body_bytes[i].is_ascii_digit() || body_bytes[i] == b'-')
        {
            i += 1;
        }
        let vtxt = body[vstart..i].trim();
        let v: u8 = vtxt
            .parse()
            .unwrap_or_else(|_| panic!("键 {:?} 的值 {:?} 不是 0-255 整数", kbuf, vtxt));
        out.insert(kbuf, v);
    }
    out
}

fn load_table() -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("data")
        .join("feat_table.json");
    fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "读不到 {:?}: {}\n请先运行: python scripts/46_export_feat_table.py",
            path, e
        )
    })
}

#[test]
fn test_chinese_table_parity() {
    let json = load_table();
    let py = parse_string_int_map(&json, "zh");
    assert!(!py.is_empty(), "中文表为空");

    let mut mismatches = Vec::new();
    for (tag, expected) in &py {
        let got = pos_to_feat_zh(tag).as_u8();
        if got != *expected {
            mismatches.push(format!(
                "  {:<10}  Python=0x{:X}  Rust=0x{:X}",
                tag, expected, got
            ));
        }
    }
    assert!(
        mismatches.is_empty(),
        "中文表有 {} 处不一致（Python vs Rust）:\n{}",
        mismatches.len(),
        mismatches.join("\n")
    );
    println!("✅ 中文表 {} 个标签全部一致", py.len());
}

#[test]
fn test_english_table_parity() {
    let json = load_table();
    let py = parse_string_int_map(&json, "en");
    assert!(!py.is_empty(), "英文表为空");

    let mut mismatches = Vec::new();
    for (tag, expected) in &py {
        let got = pos_to_feat_en(tag).as_u8();
        if got != *expected {
            mismatches.push(format!(
                "  {:<10}  Python=0x{:X}  Rust=0x{:X}",
                tag, expected, got
            ));
        }
    }
    assert!(
        mismatches.is_empty(),
        "英文表有 {} 处不一致（Python vs Rust）:\n{}",
        mismatches.len(),
        mismatches.join("\n")
    );
    println!("✅ 英文表 {} 个标签全部一致", py.len());
}

/// 反向检查：Rust 有、Python 没有的标签也要报出来。
///
/// 上面的测试只查「Python 有的 Rust 对不对得上」，漏掉了
/// 「Rust 多映射了什么」。这里用一个代表性标签集覆盖 16 个槽位，
/// 确认 Rust 侧每个槽位都可达。
#[test]
fn test_all_slots_mapped_in_rust() {
    let zh_tags = [
        "n", "v", "a", "d", "r", "p", "c", "u", "m", "q", "t", "x", "w", "i",
    ];
    let en_tags = [
        "NN", "VB", "JJ", "RB", "PRP", "IN", "CC", "MD", "CD", ".", "UH",
    ];
    let mut slots: Vec<u8> = zh_tags
        .iter()
        .map(|t| pos_to_feat_zh(t).as_u8())
        .chain(en_tags.iter().map(|t| pos_to_feat_en(t).as_u8()))
        .collect();
    slots.sort_unstable();
    slots.dedup();

    for slot in 0x0..=0xD {
        assert!(
            slots.contains(&slot),
            "槽位 0x{:X} 在 Rust 侧无任何标签映射到",
            slot
        );
    }
    // 0xE COMMON 由运行时词表决定，0xF FALLBACK 是默认分支
    assert_eq!(FeatureCode::COMMON.as_u8(), 0xE);
    assert_eq!(FeatureCode::FALLBACK.as_u8(), 0xF);
}

/// 两个语言的标签集不应冲突（同一字符串在两语言下含义不同是允许的，
/// 但各自的映射必须是确定的）。
#[test]
fn test_no_accidental_cross_language_collision() {
    // "q" 在中文是量词，在英文未定义 —— 这是设计如此，不是 bug
    assert_eq!(pos_to_feat_zh("q"), FeatureCode::MEASURE);
    assert_eq!(pos_to_feat_en("q"), FeatureCode::FALLBACK);

    // "NN" 在英文是名词，在中文未定义
    assert_eq!(pos_to_feat_zh("NN"), FeatureCode::FALLBACK);
    assert_eq!(pos_to_feat_en("NN"), FeatureCode::NOUN);

    // 空标签应落兜底，而不是 panic
    assert_eq!(pos_to_feat_zh(""), FeatureCode::FALLBACK);
    assert_eq!(pos_to_feat_en(""), FeatureCode::FALLBACK);
}
