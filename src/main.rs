use rust_bert::pipelines::sentiment::SentimentClassifier;
use rust_bert::resources::{RemoteResource, Resource};
use tch::Device;
use tokio;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let device = Device::Cpu;

    // Define the remote resources for the BERT model and tokenizer files
    let vocab_resource = Resource::Remote(RemoteResource::from_pretrained("bert-base-uncased-vocab.txt"));
    let config_resource = Resource::Remote(RemoteResource::from_pretrained("bert-base-uncased-config.json"));
    let weights_resource = Resource::Remote(RemoteResource::from_pretrained("bert-base-uncased-model.bin"));
    let model_resource = (vocab_resource, config_resource, weights_resource);

    // Initialize the SentimentClassifier using the remote resources
    let sentiment_classifier = SentimentClassifier::new(model_resource, device).await?;

    // Predict the sentiment of a sample sentence
    let input_text = "This is a good book!";
    let sentiment = sentiment_classifier.predict(&[input_text]);
    println!("{:?}", sentiment);

    Ok(())
}
