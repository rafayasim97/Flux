use crate::lexer::{Token, Lexer};
use crate::ast::{Expr, Stmt, Function, MatrixType};

pub struct Parser { lexer: Lexer, current_token: Token }

impl Parser {
    pub fn new(mut lexer: Lexer) -> Self { let current_token = lexer.next_token(); Parser { lexer, current_token } }
    fn eat(&mut self) { self.current_token = self.lexer.next_token(); }

    pub fn parse_function(&mut self) -> Function {
        let is_pure = match self.current_token { Token::Pure => { self.eat(); true }, _ => false };
        if let Token::Fn = self.current_token { self.eat(); } else { panic!("Expected 'fn'"); }
        let name = match &self.current_token { Token::Identifier(n) => n.clone(), _ => panic!("Func Name?") };
        self.eat();
        if let Token::LParen = self.current_token { self.eat(); }
        if let Token::RParen = self.current_token { self.eat(); }
        if let Token::LBrace = self.current_token { self.eat(); }
        let mut body = Vec::new();
        while self.current_token != Token::RBrace && self.current_token != Token::EOF { body.push(self.parse_statement()); }
        self.eat(); 
        Function { name, body, is_pure }
    }

    fn parse_statement(&mut self) -> Stmt {
        match self.current_token {
            Token::Let => {
                self.eat();
                let name = match &self.current_token { Token::Identifier(n) => n.clone(), _ => panic!("Var Name?") };
                self.eat();
                if let Token::Equals = self.current_token { self.eat(); }
                let value = self.parse_expression();
                Stmt::Let { name, value }
            },
            Token::Save => {
                self.eat();
                if let Token::LParen = self.current_token { self.eat(); }
                let matrix_name = match &self.current_token { Token::Identifier(n) => n.clone(), _ => panic!("Matrix Name?") };
                self.eat();
                if let Token::Comma = self.current_token { self.eat(); }
                let filename = match &self.current_token { Token::StringLiteral(s) => s.clone(), _ => panic!("Filename?") };
                self.eat();
                if let Token::RParen = self.current_token { self.eat(); }
                Stmt::Save { matrix_name, filename }
            },
            _ => Stmt::Expr(self.parse_expression()),
        }
    }

    fn parse_expression(&mut self) -> Expr {
        let mut left = self.parse_primary();
        while let Token::Pipe = self.current_token {
            self.eat(); 
            let function_name = match &self.current_token {
                Token::Identifier(n) => n.clone(),
                Token::Mse => "mse".to_string(),
                Token::Relu => "relu".to_string(),
                _ => panic!("Expected function after |>"),
            };
            self.eat();
            let mut args = Vec::new();
            if let Token::LParen = self.current_token {
                self.eat();
                if let Token::Identifier(n) = &self.current_token { args.push(Expr::Variable(n.clone())); self.eat(); }
                if let Token::RParen = self.current_token { self.eat(); }
            }
            left = Expr::Pipe { left: Box::new(left), function: function_name, args };
        }
        left
    }

    fn parse_primary(&mut self) -> Expr {
        match &self.current_token {
            Token::Number(n) => { let v = *n; self.eat(); Expr::Number(v) },
            Token::StringLiteral(s) => { let v = s.clone(); self.eat(); Expr::StringLiteral(v) },
            Token::Matrix => self.parse_matrix(MatrixType::F32),
            Token::MatrixBf16 => self.parse_matrix(MatrixType::Bf16),
            Token::Load => self.parse_load(), // NEW
            Token::Identifier(s) => { let v = s.clone(); self.eat(); Expr::Variable(v) },
            _ => panic!("Unexpected token: {:?}", self.current_token),
        }
    }

    fn parse_matrix(&mut self, dtype: MatrixType) -> Expr {
        self.eat(); 
        if let Token::LParen = self.current_token { self.eat(); }
        let rows = match self.current_token { Token::Number(n) => n as usize, _ => panic!("Row count?") };
        self.eat();
        if let Token::Comma = self.current_token { self.eat(); }
        let cols = match self.current_token { Token::Number(n) => n as usize, _ => panic!("Col count?") };
        self.eat();
        if let Token::RParen = self.current_token { self.eat(); }
        Expr::Matrix { rows, cols, dtype }
    }

    fn parse_load(&mut self) -> Expr {
        self.eat();
        if let Token::LParen = self.current_token { self.eat(); }
        let rows = match self.current_token { Token::Number(n) => n as usize, _ => panic!("Row count?") };
        self.eat();
        if let Token::Comma = self.current_token { self.eat(); }
        let cols = match self.current_token { Token::Number(n) => n as usize, _ => panic!("Col count?") };
        self.eat();
        if let Token::Comma = self.current_token { self.eat(); }
        let filename = match &self.current_token { Token::StringLiteral(s) => s.clone(), _ => panic!("Filename?") };
        self.eat();
        if let Token::RParen = self.current_token { self.eat(); }
        Expr::Load { rows, cols, filename }
    }
}