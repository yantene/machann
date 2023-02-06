use rust_bert::gpt2::GPT2Generator;
use rust_bert::pipelines::common::{ModelType, TokenizerOption};
use rust_bert::pipelines::generation_utils::{GenerateConfig, LanguageGenerator};
use rust_bert::resources::LocalResource;
use std::path::PathBuf;

use crate::chatbot::models::responder_intelligence::ResponderIntelligence;

pub struct JapaneseGpt1bResponder {
    generator: GPT2Generator,
}

impl ResponderIntelligence for JapaneseGpt1bResponder {
    fn new() -> Self {
        let model_path = "./models/japanese-gpt-1b/rust_model.ot";
        let model_resource = Box::new(LocalResource {
            local_path: PathBuf::from(model_path),
        });

        let config_path = "./models/japanese-gpt-1b/config.json";
        let config_resource = Box::new(LocalResource {
            local_path: PathBuf::from(config_path),
        });

        let vocab_path = "./models/japanese-gpt-1b/spiece.model";
        let vocab_resource = Box::new(LocalResource {
            // Open spiece.model in a binary editor and replace the data as follows.
            // [CLS] -> <cls>, [UNK] -> <unk>, [PAD] -> <pad>, [SEP] -> <sep>, [MASK] -> <mask>
            local_path: PathBuf::from(vocab_path),
        });
        let merges_resource = vocab_resource.clone();

        let generate_config = GenerateConfig {
            model_resource,
            config_resource,
            vocab_resource,
            merges_resource: Some(merges_resource),
            top_k: 500,
            ..Default::default()
        };

        let tokenizer_option =
            TokenizerOption::from_file(ModelType::T5, vocab_path, None, false, None, None).unwrap();

        let gpt2_generator =
            GPT2Generator::new_with_tokenizer(generate_config, tokenizer_option).unwrap();

        Self {
            generator: gpt2_generator,
        }
    }

    fn respond(&self, request: &str) -> String {
        let preprocessed_query = request
            .replace('\n', "")
            .replace('「', "『")
            .replace('」', "』");

        let input_text = format!("わたし「{preprocessed_query}」マサル「");

        let generated = self.generator.generate(Some(&[&input_text]), None);

        let output_text = &generated[0].text;

        output_text
            .replace(['「', '」'], "\0")
            .split('\0')
            .nth(3)
            .unwrap()
            .to_string()
    }
}
