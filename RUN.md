Running the Sentiment Analysis 


1. Install Rust and Cargo. https://www.rust-lang.org/tools/install
2. Create a new Rust project using cargo new.
3. Copy the main.rs code.
4. cargo run


Note: In this project, we are not using bert.rs, sentiment.rs, and tokenizer.rs in the src folder. Since to be able to test out this code, we need a bert model directory that typically has vocab text. But we were not able to create it. Thus, we are using in-built pipelines from rust-bert for the model to work in main.rs
