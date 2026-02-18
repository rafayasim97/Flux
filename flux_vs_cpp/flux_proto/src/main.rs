mod lexer; mod ast; mod parser; mod codegen;
use lexer::Lexer; use parser::Parser; use codegen::CodeGen;

fn main() {
    // 0 = Auto-Detect rows from file size
    let code = r#"
        pure fn train_flux_alphabet() {
            // Data (90 Columns = 64 Time + 26 Char Flags)
            let Input = load(0, 90, "input_alphabet.bin")
            let Target = load(0, 80, "target_alphabet.bin")
            
            // Weights (90 Input -> 512 Hidden -> 80 Output)
            let W1 = matrix_bf16(90, 512)
            let W2 = matrix_bf16(512, 80)

            // Train (Adam will handle the optimization)
            Input |> dot(W1) |> relu() |> dot(W2) |> mse(Target)
            
            // Save weights for the TTS engine
            save(W1, "hello_w1.bin")
            save(W2, "hello_w2.bin")
        }
    "#;

    eprintln!("--- FLUX ALPHABET TRAINING ---");
    let lexer = Lexer::new(code);
    let mut parser = Parser::new(lexer);
    let ast = parser.parse_function();
    let codegen = CodeGen::new();
    println!("{}", codegen.generate(&ast));
}