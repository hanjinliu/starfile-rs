use pyo3::prelude::*;

#[pyclass]
pub struct DataBlock {
    pub name: String,
    pub block_type: BlockData,
}

#[pyclass]
pub enum BlockType {
    Simple,
    Loop,
}

#[pymethods]
impl BlockType {
    pub fn is_loop(&self) -> bool {
        matches!(self, BlockType::Loop)
    }
}

impl DataBlock {
    pub fn new(name: String, block_type: BlockData) -> Self {
        DataBlock { name, block_type }
    }
}

#[pymethods]
impl DataBlock {
    #[staticmethod]
    pub fn construct_single_block(name: String, scalars: Vec<(String, String)>) -> Self {
        let scalars = scalars
            .into_iter()
            .map(|(n, v)| Scalar::new(n, v))
            .collect();
        DataBlock::new(name.clone(), BlockData::Simple(scalars))
    }

    #[staticmethod]
    pub fn construct_loop_block(
        name: String,
        columns: Vec<String>,
        content: String,
        nrows: usize,
    ) -> Self {
        let loop_data = LoopData::new(columns, content, nrows);
        DataBlock::new(name.clone(), BlockData::Loop(loop_data))
    }

    /// Get the name of the data block
    pub fn name(&self) -> String {
        self.name.clone()
    }

    pub fn set_name(&mut self, name: String) {
        self.name = name;
    }

    pub fn column_names(&self) -> PyResult<Vec<String>> {
        match &self.block_type {
            BlockData::Simple(scalars) => {
                let names = scalars.iter().map(|s| s.name.clone()).collect();
                Ok(names)
            }
            BlockData::Loop(loop_data) => {
                Ok(loop_data.columns.clone())
            }
            BlockData::EOF => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "EOF block has no columns",
            )),
        }
    }

    pub fn set_column_names(&mut self, names: Vec<String>) -> PyResult<()> {
        match &mut self.block_type {
            BlockData::Loop(loop_data) => {
                if names.len() != loop_data.columns.len() {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Number of names does not match number of columns.",
                    ));
                }
                loop_data.columns = names;
                Ok(())
            }
            BlockData::Simple(simple_data) => {
                if names.len() != simple_data.len() {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Number of names does not match number of scalars.",
                    ));
                }
                for (scalar, name) in simple_data.iter_mut().zip(names.iter()) {
                    scalar.name = name.clone();
                }
                Ok(())
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Not a loop data block",
            )),
        }
    }

    pub fn single_to_list(&self) -> PyResult<Vec<(String, String)>> {
        match &self.block_type {
            BlockData::Simple(scalars) => {
                let mut result = Vec::new();
                for scalar in scalars {
                    result.push((scalar.name.clone(), scalar.value.clone()));
                }
                Ok(result)
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Not a simple data block",
            )),
        }
    }

    pub fn loop_columns(&self) -> PyResult<Vec<String>> {
        match &self.block_type {
            BlockData::Loop(loop_data) => {
                Ok(loop_data.columns.clone())
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Not a loop data block",
            )),
        }
    }

    pub fn loop_content(&self) -> PyResult<&String> {
        match &self.block_type {
            BlockData::Loop(loop_data) => {
                Ok(&loop_data.content)
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Not a loop data block",
            )),
        }
    }

    pub fn loop_content_with_sep(&self, sep: &str) -> PyResult<String> {
        match &self.block_type {
            BlockData::Loop(loop_data) => {
                let lines = loop_data.content
                    .lines()
                    .map(|line| line.split_whitespace().collect::<Vec<&str>>().join(sep))
                    .collect::<Vec<String>>();
                Ok(lines.join("\n"))
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Not a loop data block",
            )),
        }
    }

    pub fn loop_nrows(&self) -> PyResult<usize> {
        match &self.block_type {
            BlockData::Loop(loop_data) => {
                Ok(loop_data.nrows)
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Not a loop data block",
            )),
        }
    }

    pub fn as_single(&self) -> PyResult<DataBlock> {
        match &self.block_type {
            BlockData::Loop(loop_data) => {
                if loop_data.nrows != 1 {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Cannot convert loop data block with multiple rows to single data block.",
                    ));
                }
                // trust there is only one line
                let first_line = loop_data.content.lines().next().unwrap();
                let values = first_line.split_whitespace().collect::<Vec<&str>>();
                // convert to scalars
                let mut scalars = Vec::new();
                for (name, value) in loop_data.columns.iter().zip(values.iter()) {
                    scalars.push(Scalar::new(name.clone(), value.to_string()));
                }
                Ok(DataBlock::new(self.name.clone(), BlockData::Simple(scalars)))
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Not a loop data block",
            )),
        }
    }

    pub fn as_loop(&self) -> PyResult<DataBlock> {
        match &self.block_type {
            BlockData::Simple(scalars) => {
                let columns = scalars.iter().map(|s| s.name.clone()).collect();
                let content = scalars.iter().map(|s| s.value.clone()).collect::<Vec<String>>().join(" ");
                let loop_data = LoopData::new(columns, content, 1);
                Ok(DataBlock::new(self.name.clone(), BlockData::Loop(loop_data)))
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Not a simple data block",
            )),
        }
    }

    pub fn block_type(&self) -> BlockType {
        match &self.block_type {
            BlockData::EOF => panic!("EOF block has no type"),
            BlockData::Simple(_) => BlockType::Simple,
            BlockData::Loop(_) => BlockType::Loop,
        }
    }

    pub fn append_values_to_loop(&mut self, values: Vec<String>) -> PyResult<()> {
        match &mut self.block_type {
            BlockData::Loop(loop_data) => {
                if values.len() != loop_data.columns.len() {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Number of values does not match number of columns.",
                    ));
                }
                let line = values.join(" ");
                if !loop_data.content.is_empty() {
                    loop_data.content.push('\n');
                }
                loop_data.content.push_str(&line);
                loop_data.nrows += 1;
                Ok(())
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Not a loop data block",
            )),
        }
    }
}

pub enum BlockData {
    EOF,
    Simple(Vec<Scalar>),
    Loop(LoopData),
}

impl BlockData {
    pub fn is_eof(&self) -> bool {
        matches!(self, BlockData::EOF)
    }
}

pub struct LoopData{
    columns: Vec<String>,
    content: String,
    nrows: usize,
}

impl LoopData {
    pub fn new(columns: Vec<String>, content: String, nrows: usize) -> Self {
        LoopData { columns, content, nrows }
    }
}

#[derive(Clone)]
pub struct Scalar {
    name: String,
    value: String,
}

impl Scalar {
    pub fn new(name: String, value: String) -> Self {
        Scalar { name, value }
    }

    /// Parse a line to a Scalar
    pub fn from_line(line: &str) -> Self {
        let parts: Vec<&str> = line.splitn(2, char::is_whitespace).collect();
        if parts.len() >= 2 {
            let val_str = parts[1].trim();
            // trim quotes if present
            let val_str_normed = if val_str.starts_with('"') && val_str.ends_with('"') {
                &val_str[1..val_str.len()-1]
            } else if val_str.starts_with('\'') && val_str.ends_with('\'') {
                &val_str[1..val_str.len()-1]
            } else {
                val_str
            };
            Scalar::new(parts[0][1..].to_string(), val_str_normed.to_string())
        } else {
            Scalar::new(parts[0][1..].to_string(), String::new())
        }
    }

}
