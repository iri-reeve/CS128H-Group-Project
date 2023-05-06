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
//but this doesn't use tokenizer.rs, bert.rs, and sentiment.rs. To integrate these files in main.rs, 
//we need a bert model directory that stores the vocab files and pre-trained bert model.
//However, I was not able to create that. So, I am using the rust-bert and pipelines that allow the code to be run.
// The typical bert model's files include the model weights, configuration file, and tokenizer vocab file.
