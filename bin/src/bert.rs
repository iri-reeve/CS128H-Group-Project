use rust_bert::bert::{BertConfig, BertForSequenceClassification};
use rust_bert::resources::{RemoteResource, Resource};
use rust_bert::Config;
use tch::{nn, Device, Tensor};

pub struct BertModel {
    model: BertForSequenceClassification,
    device: Device,
}

impl BertModel {
    pub fn from_pretrained(model_path: &str) -> Self {
        let device = Device::cuda_if_available();
        let config_path = format!("{}/config.json", model_path);
        let vocab_path = format!("{}/vocab.txt", model_path);
        let weights_path = format!("{}/pytorch_model.bin", model_path);

        let config_resource = Resource::LocalFile {
            local_path: config_path.into(),
        };
        let vocab_resource = Resource::LocalFile {
            local_path: vocab_path.into(),
        };
        let weights_resource = Resource::LocalFile {
            local_path: weights_path.into(),
        };

        let config = BertConfig::from_resource(&config_resource);
        let bert_model = BertForSequenceClassification::from_pretrained(
            &config,
            &weights_resource,
            &vocab_resource,
            false,
            false,
            None,
        )
        .unwrap();

        BertModel {
            model: bert_model,
            device,
        }
    }

    pub fn forward(&self, input: &[i64]) -> Vec<f32> {
        let input_tensor = Tensor::of_slice(input).to(self.device);
        let (logits, _) = self.model.forward_t(Some(&input_tensor), None, None, None, false);
        logits.detach().to_device(Device::Cpu).to_vec::<f32>().unwrap()
    }
}
