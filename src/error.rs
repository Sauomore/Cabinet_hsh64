use std::fmt;

/// HSH-64 编码层错误类型
#[derive(Debug, Clone, PartialEq)]
pub enum EncodeError {
    /// 未知词性标签
    UnknownPOSTag(String),
    /// 相似码超出 52-bit 范围
    Sim64OutOfRange,
    /// 绝对码分配溢出
    AbsOverflow,
    /// IO 错误
    Io(String),
    /// 配置错误
    Config(String),
}

impl fmt::Display for EncodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EncodeError::UnknownPOSTag(tag) => write!(f, "未知词性标签: {}", tag),
            EncodeError::Sim64OutOfRange => write!(f, "相似码超出 52-bit 范围"),
            EncodeError::AbsOverflow => write!(f, "绝对码分配溢出"),
            EncodeError::Io(msg) => write!(f, "IO 错误: {}", msg),
            EncodeError::Config(msg) => write!(f, "配置错误: {}", msg),
        }
    }
}

impl std::error::Error for EncodeError {}

impl From<std::io::Error> for EncodeError {
    fn from(e: std::io::Error) -> Self {
        EncodeError::Io(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let e = EncodeError::UnknownPOSTag("xyz".to_string());
        assert_eq!(e.to_string(), "未知词性标签: xyz");
    }
}
