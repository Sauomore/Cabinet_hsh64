//! 纯 HSH-64 基准测试
//!
//! 用法：
//!   cargo run --example benchmark_pure_hsh64 -- \
//!       --embedding tests/data/embedding.cache \
//!       --pca tests/data/pca_52.bin \
//!       --top-k 10 \
//!       --queries 100

use hsh64::{
    embedding::{EmbeddingModel, FileCachedEmbedding}, encoder::EncoderConfig,
    mih_index::MihSemanticIndex, pca::PcaProjection,
};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

#[derive(Debug, Clone)]
struct Args {
    embedding: PathBuf,
    pca: PathBuf,
    top_k: usize,
    queries: usize,
    segment_counts: Vec<usize>,
    radii: Vec<u32>,
}

fn parse_args() -> Args {
    let mut args = std::env::args().skip(1);
    let mut embedding = PathBuf::from("tests/data/embedding.cache");
    let mut pca = PathBuf::from("tests/data/pca_52.bin");
    let mut top_k = 10usize;
    let mut queries = 100usize;
    let mut segment_counts = vec![1usize, 2, 4, 13];
    let mut radii = vec![0u32, 2, 4, 6, 8];

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--embedding" => embedding = PathBuf::from(args.next().expect("--embedding 需要值")),
            "--pca" => pca = PathBuf::from(args.next().expect("--pca 需要值")),
            "--top-k" => top_k = args.next().expect("--top-k 需要值").parse().expect("top-k 必须是整数"),
            "--queries" => queries = args.next().expect("--queries 需要值").parse().expect("queries 必须是整数"),
            "--segment-counts" => {
                let s = args.next().expect("--segment-counts 需要值");
                segment_counts = s.split(',').map(|x| x.parse().unwrap()).collect();
            }
            "--radii" => {
                let s = args.next().expect("--radii 需要值");
                radii = s.split(',').map(|x| x.parse().unwrap()).collect();
            }
            _ => {}
        }
    }

    Args {
        embedding,
        pca,
        top_k,
        queries,
        segment_counts,
        radii,
    }
}

/// 计算余弦相似度
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na < 1e-8 || nb < 1e-8 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// 获取真实 top-K 邻居
fn ground_truth_topk(
    query_idx: usize,
    vectors: &[Vec<f32>],
    k: usize,
) -> Vec<usize> {
    let query = &vectors[query_idx];
    let mut scores: Vec<(usize, f32)> = vectors
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != query_idx)
        .map(|(i, v)| (i, cosine_similarity(query, v)))
        .collect();
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scores.into_iter().take(k).map(|(i, _)| i).collect()
}

/// 计算 Recall@K
fn recall_at_k(retrieved: &[String], truth_words: &HashSet<String>, k: usize) -> f32 {
    let k = k.min(truth_words.len());
    if k == 0 {
        return 0.0;
    }
    let retrieved_set: HashSet<String> = retrieved.iter().take(k).cloned().collect();
    let hits = retrieved_set.intersection(truth_words).count();
    hits as f32 / k as f32
}

fn main() {
    let args = parse_args();
    println!("HSH-64 纯索引基准测试");
    println!("embedding: {}", args.embedding.display());
    println!("pca:       {}", args.pca.display());
    println!("top-k:     {}", args.top_k);
    println!("queries:   {}", args.queries);
    println!();

    // 1. 加载 embedding
    let t0 = Instant::now();
    let embedding = Arc::new(
        FileCachedEmbedding::new(&args.embedding, 64).expect("加载 embedding 失败"),
    );
    let vocab = embedding.vocab();
    let vectors: Vec<Vec<f32>> = vocab.iter().map(|w| embedding.embed(w)).collect();
    let dim = embedding.dim();
    println!("[加载] {} 个 {}-dim 向量，耗时 {:.2?}", vocab.len(), dim, t0.elapsed());

    // 2. 加载 PCA
    let t0 = Instant::now();
    let pca_bytes = std::fs::read(&args.pca).expect("读取 PCA 失败");
    let pca = PcaProjection::from_bytes(&pca_bytes).expect("解析 PCA 失败");
    println!("[加载 PCA] n_components={}, 耗时 {:.2?}", pca.n_components, t0.elapsed());

    // 3. 构建 Encoder
    let t0 = Instant::now();
    let config = EncoderConfig {
        embed_dim: dim,
        embedding_cache_path: Some(args.embedding.clone()),
        pca_path: Some(args.pca.clone()),
        ..Default::default()
    };
    let encoder = hsh64::Encoder::with_config(config).expect("创建 Encoder 失败");
    println!("[构建 Encoder] 耗时 {:.2?}", t0.elapsed());

    // 4. 预计算真实邻居
    let n_queries = args.queries.min(vocab.len());
    let query_indices: Vec<usize> = (0..n_queries).collect();
    let mut truth_map: HashMap<String, HashSet<String>> = HashMap::new();
    for &idx in &query_indices {
        let topk = ground_truth_topk(idx, &vectors, args.top_k);
        let truth: HashSet<String> = topk.into_iter().map(|i| vocab[i].clone()).collect();
        truth_map.insert(vocab[idx].clone(), truth);
    }

    // 5. 编码所有词并统计唯一 sim 码比例
    let t0 = Instant::now();
    let mut unique_sims = HashSet::new();
    for word in &vocab {
        let code = encoder.encode_word_with_pos(word, "n");
        unique_sims.insert(code.sim());
    }
    println!(
        "[编码词表] {} 个唯一 sim 码 / {} 个词 = {:.2}，耗时 {:.2?}",
        unique_sims.len(),
        vocab.len(),
        unique_sims.len() as f32 / vocab.len() as f32,
        t0.elapsed()
    );

    // 6. 暴力 Hamming 基准
    println!("\n========== 暴力 Hamming 扫描 ==========");
    let t0 = Instant::now();
    let mut total_recall = 0.0f32;
    for &idx in &query_indices {
        let query_word = &vocab[idx];
        let query_code = encoder.encode_word_with_pos(query_word, "n");
        let mut scored: Vec<(String, u32)> = vocab
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != idx)
            .map(|(_, w)| {
                let code = encoder.encode_word_with_pos(w, "n");
                (w.clone(), code.sim_hamming_distance(&query_code))
            })
            .collect();
        scored.sort_by(|a, b| a.1.cmp(&b.1));
        let retrieved: Vec<String> = scored.into_iter().take(args.top_k).map(|(w, _)| w).collect();
        total_recall += recall_at_k(&retrieved, truth_map.get(query_word).unwrap(), args.top_k);
    }
    let elapsed = t0.elapsed();
    let qps = n_queries as f32 / elapsed.as_secs_f32();
    println!(
        "Recall@{} = {:.4}, QPS = {:.1}, 总耗时 {:.2?}",
        args.top_k,
        total_recall / n_queries as f32,
        qps,
        elapsed
    );

    // 7. MIH 多索引哈希基准
    println!("\n========== MIH 多索引哈希 ==========");
    for &segment_count in &args.segment_counts {
        if 52 % segment_count != 0 {
            println!("跳过 segment_count={}（不能整除 52）", segment_count);
            continue;
        }

        let t0 = Instant::now();
        let index = MihSemanticIndex::build_with_embedding(
            encoder.clone(),
            embedding.clone(),
            segment_count,
            false,
        )
        .expect("构建 MIH 索引失败");
        let build_time = t0.elapsed();

        for &radius in &args.radii {
            let t0 = Instant::now();
            let mut total_recall = 0.0f32;
            let mut total_candidates = 0usize;
            let mut total_hamming = 0u32;
            for &idx in &query_indices {
                let query_word = &vocab[idx];
                let results = index.search(query_word, args.top_k, radius, usize::MAX).unwrap();
                let retrieved: Vec<String> = results.iter().map(|r| r.word.clone()).collect();
                total_recall += recall_at_k(&retrieved, truth_map.get(query_word).unwrap(), args.top_k);
                total_candidates += retrieved.len();
                total_hamming += results.iter().map(|r| r.hamming).sum::<u32>();
            }
            let elapsed = t0.elapsed();
            let qps = n_queries as f32 / elapsed.as_secs_f32();
            println!(
                "seg={:2}, radius={:2}: Recall@{}={:.4}, 平均候选={:.1}, 平均Hamming={:.2}, QPS={:.1}, 构建={:.2?}",
                segment_count,
                radius,
                args.top_k,
                total_recall / n_queries as f32,
                total_candidates as f32 / n_queries as f32,
                total_hamming as f32 / total_candidates.max(1) as f32,
                qps,
                build_time
            );
        }
    }

    println!("\n基准测试完成");
}
