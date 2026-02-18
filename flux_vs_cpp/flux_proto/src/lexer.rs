#[derive(Debug, PartialEq, Clone)]
pub enum Token {
    Fn, Let, Pure, Matrix, MatrixBf16, Save, Load, // NEW
    Mse, Relu,
    Identifier(String), Number(f64), StringLiteral(String),
    Comma, Plus, Equals, Pipe, LParen, RParen, LBrace, RBrace, EOF,
}

pub struct Lexer { input: Vec<char>, pos: usize }

impl Lexer {
    pub fn new(input: &str) -> Self { Lexer { input: input.chars().collect(), pos: 0 } }

    pub fn next_token(&mut self) -> Token {
        self.skip_whitespace();
        if self.pos >= self.input.len() { return Token::EOF; }
        let current = self.input[self.pos];
        match current {
            '+' => { self.pos += 1; Token::Plus },
            '=' => { self.pos += 1; Token::Equals },
            '(' => { self.pos += 1; Token::LParen },
            ')' => { self.pos += 1; Token::RParen },
            '{' => { self.pos += 1; Token::LBrace },
            '}' => { self.pos += 1; Token::RBrace },
            ',' => { self.pos += 1; Token::Comma },
            '"' => self.read_string(),
            '/' => {
                if self.pos + 1 < self.input.len() && self.input[self.pos+1] == '/' {
                    self.skip_comment(); self.next_token()
                } else { panic!("Unexpected char: /"); }
            },
            '|' => {
                if self.pos + 1 < self.input.len() && self.input[self.pos+1] == '>' {
                    self.pos += 2; Token::Pipe
                } else { panic!("Unexpected char: |"); }
            },
            '0'..='9' => self.read_number(),
            'a'..='z' | 'A'..='Z' | '_' => self.read_identifier(),
            _ => panic!("Unknown character: {}", current),
        }
    }
    fn skip_whitespace(&mut self) { while self.pos < self.input.len() && self.input[self.pos].is_whitespace() { self.pos += 1; } }
    fn skip_comment(&mut self) { while self.pos < self.input.len() && self.input[self.pos] != '\n' { self.pos += 1; } }
    fn read_string(&mut self) -> Token {
        self.pos += 1; let start = self.pos;
        while self.pos < self.input.len() && self.input[self.pos] != '"' { self.pos += 1; }
        let text: String = self.input[start..self.pos].iter().collect(); self.pos += 1;
        Token::StringLiteral(text)
    }
    fn read_identifier(&mut self) -> Token {
        let start = self.pos;
        while self.pos < self.input.len() && (self.input[self.pos].is_alphanumeric() || self.input[self.pos] == '_') { self.pos += 1; }
        let text: String = self.input[start..self.pos].iter().collect();
        match text.as_str() {
            "fn" => Token::Fn, "let" => Token::Let, "pure" => Token::Pure,
            "matrix" => Token::Matrix, "matrix_bf16" => Token::MatrixBf16,
            "save" => Token::Save, "load" => Token::Load, // NEW
            "mse" => Token::Mse, "relu" => Token::Relu,
            _ => Token::Identifier(text),
        }
    }
    fn read_number(&mut self) -> Token {
        let start = self.pos;
        while self.pos < self.input.len() && (self.input[self.pos].is_numeric() || self.input[self.pos] == '.') { self.pos += 1; }
        let text: String = self.input[start..self.pos].iter().collect();
        Token::Number(text.parse().unwrap())
    }
}