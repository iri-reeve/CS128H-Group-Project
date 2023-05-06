use std::path::Path;
use tokenizers::tokenizer::{BertTokenizer, Tokenizer};
use tokenizers::vocab::Vocab;
use tokenizers::models::bpe::BPE;

pub struct BertTextTokenizer {
    tokenizer: Tokenizer<BertTokenizer>,
}

impl BertTextTokenizer {
    pub fn from_pretrained<P: AsRef<Path>>(vocab_path: P) -> Self {
        let vocab = Vocab::from_file(vocab_path.as_ref().to_str().unwrap()).unwrap();
        let bpe = BPE::new(vocab);
        let tokenizer = BertTokenizer::new(bpe);

        BertTextTokenizer {
            tokenizer: Tokenizer::new(tokenizer),
        }
    }

    pub fn tokenize(&self, input_text: &str) -> Vec<String> {
        self.tokenizer.encode(input_text, true).unwrap().tokens
    }

    pub fn convert_tokens_to_ids(&self, tokens: Vec<String>) -> Vec<i64> {
        tokens
            .iter()
            .map(|token| self.tokenizer.token_to_id(token))
            .collect()
    }
}
