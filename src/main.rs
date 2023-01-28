use rust_bert::pipelines::generation_utils::LanguageGenerator;
use rust_bert::{gpt2::GPT2Generator, pipelines::generation_utils::GenerateOptions};

fn main() {
    let model = GPT2Generator::new(Default::default()).unwrap();

    let input_context_1 = "The dog";
    let input_context_2 = "The cat was";

    let generate_options = GenerateOptions {
        max_length: Some(30),
        ..Default::default()
    };

    let output = model.generate(
        Some(&[input_context_1, input_context_2]),
        Some(generate_options),
    );

    for item in output {
        println!("{}", item.text)
    }
}
