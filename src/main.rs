use std::io;

use chatbot::models::responder_intelligence::ResponderIntelligence;

mod chatbot;

fn main() -> io::Result<()> {
    let intelligence = chatbot::intelligences::japanese_gpt_1b::japanese_gpt_1b_responder::JapaneseGpt1bResponder::new();

    let mut question = String::new();

    loop {
        let bytes = io::stdin().read_line(&mut question)?;

        if bytes == 0 {
            break;
        }

        let answer = intelligence.respond(question.as_str());

        println!("{answer}");
    }

    Ok(())
}
