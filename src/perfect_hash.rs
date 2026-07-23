//! 准完美哈希实现
//!
//! 对每个簇 (feat, sim) 搜索种子 s ∈ [0, 255]，使得：
//! abs(wi) = (BKDR(wi) XOR s) mod 256
//! 对所有词 wi 互不相同。

use std::collections::HashMap;

/// BKDR 字符串哈希函数
pub fn bkdr_hash(s: &str) -> u64 {
    let seed: u64 = 131;
    let mut hash: u64 = 0;
    for byte in s.bytes() {
        hash = hash.wrapping_mul(seed).wrapping_add(byte as u64);
    }
    hash
}

/// 计算候选绝对码
pub fn compute_abs(word: &str, seed: u8) -> u8 {
    let h = bkdr_hash(word);
    ((h ^ (seed as u64)) % 256) as u8
}

/// 单个簇的完美哈希种子搜索
pub fn search_seed(words: &[String]) -> (u8, HashMap<String, u8>, Vec<u8>) {
    let n = words.len();
    if n == 0 {
        return (0, HashMap::new(), Vec::new());
    }
    if n > 256 {
        return search_best_seed(words);
    }

    for seed in 0..=255u8 {
        let mut used = [false; 256];
        let mut collision = false;
        for word in words {
            let abs = compute_abs(word, seed);
            if used[abs as usize] {
                collision = true;
                break;
            }
            used[abs as usize] = true;
        }
        if !collision {
            let mut map = HashMap::with_capacity(n);
            for word in words {
                let abs = compute_abs(word, seed);
                map.insert(word.clone(), abs);
            }
            let abs_list: Vec<u8> = words.iter().map(|w| map[w]).collect();
            return (seed, map, abs_list);
        }
    }

    search_best_seed(words)
}

fn search_best_seed(words: &[String]) -> (u8, HashMap<String, u8>, Vec<u8>) {
    let mut best_seed = 0u8;
    let mut min_collision = usize::MAX;
    let mut best_map = HashMap::new();

    for seed in 0..=255u8 {
        let mut counts = [0u16; 256];
        for word in words {
            let abs = compute_abs(word, seed);
            counts[abs as usize] += 1;
        }
        let collisions: usize = counts.iter().map(|&c| if c > 1 { (c - 1) as usize } else { 0 }).sum();
        if collisions < min_collision {
            min_collision = collisions;
            best_seed = seed;
            let mut map = HashMap::with_capacity(words.len());
            for word in words {
                let abs = compute_abs(word, seed);
                map.insert(word.clone(), abs);
            }
            best_map = map;
            if collisions == 0 {
                break;
            }
        }
    }

    let abs_list: Vec<u8> = words.iter().map(|w| best_map[w]).collect();
    (best_seed, best_map, abs_list)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_abs_stable() {
        let a = compute_abs("测试", 42);
        let b = compute_abs("测试", 42);
        assert_eq!(a, b);
    }

    #[test]
    fn test_search_seed_perfect() {
        let words: Vec<String> = (0..20).map(|i| format!("词{}", i)).collect();
        let (_, map, _) = search_seed(&words);
        assert_eq!(map.len(), words.len());
        let unique: std::collections::HashSet<_> = map.values().cloned().collect();
        assert_eq!(unique.len(), words.len());
    }
}
