use crate::bert::BertModel;
use crate::tokenizer::BertTextTokenizer;

pub enum Sentiment {
    Positive,
    Negative,
}

pub struct SentimentAnalysis {
    model: BertModel,
    tokenizer: BertTextTokenizer,
}

impl SentimentAnalysis {
    pub fn new(model_path: &str, vocab_path: &str) -> Self {
        let model = BertModel::from_pretrained(model_path);
        let tokenizer = BertTextTokenizer::from_pretrained(vocab_path);

        SentimentAnalysis { model, tokenizer }
    }

    pub fn predict(&self, input_text: &str) -> Sentiment {
        let tokens = self.tokenizer.tokenize(input_text);
        let input = self.tokenizer.convert_tokens_to_ids(tokens);
        let logits = self.model.forward(&input);
        let sentiment = self.classify_sentiment(logits);
        sentiment
    }

    fn classify_sentiment(&self, logits: Vec<f32>) -> Sentiment {
        // the classification logic based on the logits returned by the model.
        // Using a softmax function followed by selecting the class with the highest probability.
        let softmax = self.softmax(logits);
        if softmax[0] > softmax[1] {
            Sentiment::Negative
        } else {
            Sentiment::Positive
        }
    }

    fn softmax(&self, logits: Vec<f32>) -> Vec<f32> {
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = logits
            .iter()
            .map(|value| (value - max).exp())
            .sum();

        logits
            .into_iter()
            .map(|value| ((value - max).exp()) / exp_sum)
            .collect()
    }
}
