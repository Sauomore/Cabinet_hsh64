//! HSH-64 端到端集成测试
//!
//! 运行前需要先执行 scripts/end_to_end_test.py 生成测试数据：
//!   F:\python311\python.exe scripts/end_to_end_test.py

use hsh64::{Encoder, EncoderConfig, MihSemanticIndex};
use std::path::PathBuf;
use std::sync::Arc;

#[test]
fn test_end_to_end_with_pca_cache() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let data_dir = root.join("tests").join("data");

    // 如果测试数据不存在则跳过（避免 CI 失败）
    if !data_dir.join("embedding.cache").exists() {
        eprintln!("跳过：未找到 embedding.cache，请先运行 scripts/end_to_end_test.py");
        return;
    }

    let config = EncoderConfig {
        embed_dim: 64,
        embedding_cache_path: Some(data_dir.join("embedding.cache")),
        pca_path: Some(data_dir.join("pca_52.bin")),
        ..Default::default()
    };

    let encoder = Encoder::with_config(config).expect("创建 Encoder 失败");
    let embedding = Arc::new(
        hsh64::embedding::FileCachedEmbedding::new(data_dir.join("embedding.cache"), encoder.embed_dim())
            .expect("加载 embedding 失败"),
    );

    let index = MihSemanticIndex::build_with_embedding(encoder, embedding, 4, false)
        .expect("构建 MIH 索引失败");

    let results = index.search("苹果", 5, 8, 10).expect("搜索失败");
    assert!(!results.is_empty(), "搜索结果不应为空");
    for r in &results {
        assert!(!r.word.is_empty());
    }
}
