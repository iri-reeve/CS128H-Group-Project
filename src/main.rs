
use rust_bert::bert::{BertConfig, BertForSequenceClassification};
use rust_bert::pipelines::sentiment::SentimentClassifier;
use rust_bert::resources::{RemoteResource, Resource};
use rust_bert::Config;
use std::path::PathBuf;
use tch::Device;
use tokio;

fn main() -> anyhow::Result<()> {
    // Initialize Tokio runtime to run async code
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    rt.block_on(run())
}

async fn run() -> anyhow::Result<()> {
    // Set device to CPU (default)
    let device = Device::Cpu;

    // Configure the BERT model
    let bert_config = BertConfig::new(
        768,               
        12,                 
        12,                 
        3072,              
        2,                  
        0.1,               
        0.1,                
        0,                  
        false,              
        false,             
        false,             
        false,              
        false,             
        true,               
        "bert-base-uncased" 
    );

    let vocab_resource = Resource::Remote(RemoteResource::from_pretrained("bert-base-uncased-vocab.txt"));
    let config_resource = Resource::Remote(RemoteResource::from_pretrained("bert-base-uncased-config.json"));
    let weights_resource = Resource::Remote(RemoteResource::from_pretrained("bert-base-uncased-model.bin"));
    let model_resource = (vocab_resource, config_resource, weights_resource);

    let sentiment_classifier = SentimentClassifier::new(model_resource, &bert_config, device).await?;

    let sentence = "I like this book!";
    let sentiment = sentiment_classifier.predict(&[sentence]);
    println!("{:?}", sentiment);

    Ok(())
}
