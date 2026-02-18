#[derive(Debug, Clone, PartialEq)]
pub enum MatrixType { F32, Bf16 }

#[derive(Debug, Clone)]
pub enum Expr {
    Number(f64),
    StringLiteral(String),
    Variable(String),
    Matrix { rows: usize, cols: usize, dtype: MatrixType },
    Load { rows: usize, cols: usize, filename: String }, // NEW
    Pipe { left: Box<Expr>, function: String, args: Vec<Expr> },
}

#[derive(Debug, Clone)]
pub enum Stmt {
    Let { name: String, value: Expr },
    Save { matrix_name: String, filename: String },
    Expr(Expr),
}

#[derive(Debug, Clone)]
pub struct Function {
    pub name: String, pub body: Vec<Stmt>, pub is_pure: bool,
}