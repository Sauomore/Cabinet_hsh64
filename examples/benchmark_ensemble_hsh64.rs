//! HSH-64 多模型 Ensemble 基准测试
//!
//! 用法：
//!   cargo run --example benchmark_ensemble_hsh64 -- \
//!       --embedding tests/data/embedding.cache \
//!       --reranker tests/data/reranker_embedding.cache \
//!       --top-k 10 --queries 100 --radius 22 \
//!       --model tests/data/deep_hash_v3_64_distill_h256.bin \
//!       --override tests/data/sim_override_dh_distill.bin \
//!       --model tests/data/deep_hash_v3_64_distill_h256_s1.bin \
//!       --override tests/data/sim_override_dh_distill_s1.bin
//!
//! 可指定任意数量的 (deep_hash, sim_override) 对。

use hsh64::{
    embedding::{EmbeddingModel, FileCachedEmbedding},
    encoder::EncoderConfig,
    hsh64::hamming_distance64,
    mih_index::MihSemanticIndex,
};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

#[derive(Debug, Clone)]
struct ModelPair {
    deep_hash: PathBuf,
    sim_override: PathBuf,
}

#[derive(Debug, Clone)]
struct Args {
    embedding: PathBuf,
    reranker: Option<PathBuf>,
    ground_truth: Option<PathBuf>,
    top_k: usize,
    queries: usize,
    segment_count: usize,
    radius: u32,
    pairs: Vec<ModelPair>,
}

fn parse_args() -> Args {
    let mut args = std::env::args().skip(1);
    let mut embedding = PathBuf::from("tests/data/embedding.cache");
    let mut reranker = None;
    let mut ground_truth = None;
    let mut top_k = 10usize;
    let mut queries = 100usize;
    let mut segment_count = 13usize;
    let mut radius = 22u32;
    let mut pairs = Vec::new();
    let mut pending_deep_hash: Option<PathBuf> = None;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--embedding" => embedding = PathBuf::from(args.next().expect("--embedding 需要值")),
            "--reranker" => reranker = Some(PathBuf::from(args.next().expect("--reranker 需要值"))),
            "--ground-truth" => ground_truth = Some(PathBuf::from(args.next().expect("--ground-truth 需要值"))),
            "--top-k" => top_k = args.next().expect("--top-k 需要值").parse().expect("top-k 必须是整数"),
            "--queries" => queries = args.next().expect("--queries 需要值").parse().expect("queries 必须是整数"),
            "--segment-count" => segment_count = args.next().expect("--segment-count 需要值").parse().expect("segment-count 必须是整数"),
            "--radius" => radius = args.next().expect("--radius 需要值").parse().expect("radius 必须是整数"),
            "--model" => {
                if let Some(dh) = pending_deep_hash.take() {
                    pairs.push(ModelPair { deep_hash: dh, sim_override: PathBuf::new() });
                }
                pending_deep_hash = Some(PathBuf::from(args.next().expect("--model 需要值")));
            }
            "--override" => {
                let ov = PathBuf::from(args.next().expect("--override 需要值"));
                if let Some(dh) = pending_deep_hash.take() {
                    pairs.push(ModelPair { deep_hash: dh, sim_override: ov });
                } else {
                    panic!("--override 必须在 --model 之后");
                }
            }
            _ => {}
        }
    }

    if let Some(dh) = pending_deep_hash.take() {
        pairs.push(ModelPair { deep_hash: dh, sim_override: PathBuf::new() });
    }

    if pairs.is_empty() {
        pairs.push(ModelPair {
            deep_hash: PathBuf::from("tests/data/deep_hash_v3_64_distill_h256.bin"),
            sim_override: PathBuf::from("tests/data/sim_override_dh_distill.bin"),
        });
    }

    Args {
        embedding,
        reranker,
        ground_truth,
        top_k,
        queries,
        segment_count,
        radius,
        pairs,
    }
}

fn hamming_sort_results(mut candidates: Vec<(String, u32)>, top_k: usize) -> Vec<String> {
    candidates.sort_by(|a, b| a.1.cmp(&b.1));
    candidates.into_iter().take(top_k).map(|(w, _)| w).collect()
}

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

fn ground_truth_topk(query_idx: usize, vectors: &[Vec<f32>], k: usize) -> Vec<usize> {
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
    let pure_hsh = args.reranker.is_none();
    println!("HSH-64 Ensemble 基准测试");
    println!("embedding:   {}", args.embedding.display());
    if let Some(ref r) = args.reranker {
        println!("reranker:    {}", r.display());
    } else {
        println!("reranker:    (无，纯 HSH 模式)");
    }
    println!("top-k:       {}", args.top_k);
    println!("queries:     {}", args.queries);
    println!("segment:     {}", args.segment_count);
    println!("radius:      {}", args.radius);
    println!("models:      {}", args.pairs.len());
    for (i, p) in args.pairs.iter().enumerate() {
        println!("  [{}] deep_hash={}", i, p.deep_hash.display());
        if !p.sim_override.as_os_str().is_empty() {
            println!("       override={}", p.sim_override.display());
        }
    }
    println!();

    let t0 = Instant::now();
    let embedding: Arc<dyn EmbeddingModel> =
        Arc::new(FileCachedEmbedding::new(&args.embedding, 64).expect("加载 embedding 失败"));
    let reranker: Option<Arc<dyn EmbeddingModel>> = args.reranker.as_ref().map(|r| {
        Arc::new(FileCachedEmbedding::new(r, 0).expect("加载 reranker 失败")) as Arc<dyn EmbeddingModel>
    });
    let vocab = embedding.vocab();
    let vectors: Vec<Vec<f32>> = vocab.iter().map(|w| embedding.embed(w)).collect();
    let reranker_vectors: Vec<Vec<f32>> = if let Some(ref gt_path) = args.ground_truth {
        let gt_emb = FileCachedEmbedding::new(gt_path, 0).expect("加载 ground-truth 失败");
        vocab.iter().map(|w| gt_emb.embed(w)).collect()
    } else {
        match &reranker {
            Some(r) => vocab.iter().map(|w| r.embed(w)).collect(),
            None => vectors.clone(),
        }
    };
    println!("[加载] {} 个 {}-dim 向量，reranker {}-dim，耗时 {:.2?}", vocab.len(), vectors[0].len(), if let Some(r)=&reranker{r.dim()}else{0}, t0.elapsed());

    // 构建多个 MIH 索引
    let t0 = Instant::now();
    let mut indices = Vec::new();
    for pair in &args.pairs {
        let config = EncoderConfig {
            embed_dim: embedding.dim(),
            embedding_cache_path: Some(args.embedding.clone()),
            pca_path: Some(PathBuf::from("tests/data/pca_3109_52.bin")),
            deep_hash_path: Some(pair.deep_hash.clone()),
            sim_override_path: if pair.sim_override.as_os_str().is_empty() { None } else { Some(pair.sim_override.clone()) },
            ..Default::default()
        };
        let encoder = hsh64::Encoder::with_config(config).expect("创建 Encoder 失败");
        let index = if let Some(ref r) = reranker {
            MihSemanticIndex::build_with_reranker(encoder, embedding.clone(), r.clone(), args.segment_count, false)
                .expect("构建带 reranker 的 MIH 索引失败")
        } else {
            MihSemanticIndex::build_with_embedding(encoder, embedding.clone(), args.segment_count, false)
                .expect("构建 MIH 索引失败")
        };
        indices.push(index);
    }
    println!("[构建 {} 个 MIH 索引] 耗时 {:.2?}", indices.len(), t0.elapsed());

    // 预计算真实邻居（基于 reranker，无 reranker 时基于 embedding）
    let n_queries = args.queries.min(vocab.len());
    let query_indices: Vec<usize> = (0..n_queries).collect();
    let mut truth_map: HashMap<String, HashSet<String>> = HashMap::new();
    for &idx in &query_indices {
        let topk = ground_truth_topk(idx, &reranker_vectors, args.top_k);
        let truth: HashSet<String> = topk.into_iter().map(|i| vocab[i].clone()).collect();
        truth_map.insert(vocab[idx].clone(), truth);
    }

    // 单模型基准（第一个模型）
    if !indices.is_empty() {
        let t0 = Instant::now();
        let mut total_recall = 0.0f32;
        let mut total_candidates = 0usize;
        for &idx in &query_indices {
            let query_word = &vocab[idx];
            let results = indices[0].search(query_word, args.top_k, args.radius, usize::MAX).unwrap();
            let retrieved: Vec<String> = results.iter().map(|r| r.word.clone()).collect();
            total_recall += recall_at_k(&retrieved, truth_map.get(query_word).unwrap(), args.top_k);
            total_candidates += retrieved.len();
        }
        let elapsed = t0.elapsed();
        println!(
            "\n单模型: Recall@{}={:.4}, 平均候选={:.1}, QPS={:.1}, 耗时 {:.2?}",
            args.top_k,
            total_recall / n_queries as f32,
            total_candidates as f32 / n_queries as f32,
            n_queries as f32 / elapsed.as_secs_f32(),
            elapsed
        );
    }

    // Ensemble 基准：候选取并集，reranker 精排或纯 Hamming 排序
    let t0 = Instant::now();
    let mut total_recall = 0.0f32;
    let mut total_union = 0usize;
    let mut total_min_hamming = 0u32;
    for &idx in &query_indices {
        let query_word = &vocab[idx];

        let mut seen: HashSet<String> = HashSet::new();
        let mut union_candidates: Vec<(String, u32)> = Vec::new();

        for index in &indices {
            let query_code = index.encoder().encode_word_with_pos(query_word, "n");
            let query_sim = query_code.sim();
            let candidates = index.collect_candidates(query_sim, args.radius);
            for word in candidates {
                if seen.insert(word.clone()) {
                    let code = index.encoder().encode_word_with_pos(&word, "n");
                    let dist = hamming_distance64(code.sim(), query_sim);
                    union_candidates.push((word, dist));
                }
            }
        }

        let retrieved: Vec<String> = if pure_hsh {
            hamming_sort_results(union_candidates, args.top_k)
        } else {
            let query_rerank = reranker.as_ref().unwrap().embed(query_word);
            let mut scored: Vec<(String, f32, u32)> = union_candidates
                .into_iter()
                .map(|(word, hamming)| {
                    let v = reranker.as_ref().unwrap().embed(&word);
                    let score = cosine_similarity(&query_rerank, &v);
                    (word, score, hamming)
                })
                .collect();
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            total_min_hamming += scored.iter().map(|(_, _, h)| *h).min().unwrap_or(0);
            scored.iter().take(args.top_k).map(|(w, _, _)| w.clone()).collect()
        };
        total_recall += recall_at_k(&retrieved, truth_map.get(query_word).unwrap(), args.top_k);
        total_union += seen.len();
    }
    let elapsed = t0.elapsed();
    println!(
        "Ensemble: Recall@{}={:.4}, 平均并集={:.1}, 平均最小Hamming={:.2}, QPS={:.1}, 耗时 {:.2?}",
        args.top_k,
        total_recall / n_queries as f32,
        total_union as f32 / n_queries as f32,
        total_min_hamming as f32 / n_queries as f32,
        n_queries as f32 / elapsed.as_secs_f32(),
        elapsed
    );

    println!("\n基准测试完成");
}
