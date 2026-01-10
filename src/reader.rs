use pyo3::prelude::*;
use std::fs::File;
use std::io::{self, BufRead};
use std::usize;
use crate::blocks::{DataBlock, BlockData, Scalar, LoopData};

#[pyclass]
pub struct StarReader {
    iter: Option<StarBufIter<io::BufReader<File>>>,
}

#[pymethods]
impl StarReader {
    #[new]
    pub fn new(path: String) -> io::Result<Self> {
        let file = File::open(path)?;
        let reader = io::BufReader::new(file);
        Ok(StarReader {
            iter: Some(StarBufIter::new(reader)),
        })
    }

    pub fn next_block(&mut self) -> PyResult<Option<DataBlock>> {
        match self.iter.as_mut() {
            Some(it) => match it.next() {
                Some(Ok(block)) => Ok(Some(block)),
                Some(Err(e)) => Err(PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string())),
                None => Ok(None),
            }
            None => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Reader has been closed",
            )),
        }
    }

    // TODO: implement chunked reading
    // pub fn next_block_chunk(&mut self, n: usize) -> PyResult<Vec<DataBlock>> {}

    pub fn close(&mut self) {
        // Explicitly drop the iterator to close the file
        self.iter = None;
    }
}

#[pyclass]
pub struct StarTextReader {
    iter: Option<StarBufIter<io::BufReader<std::io::Cursor<String>>>>,
}

#[pymethods]
impl StarTextReader {
    #[new]
    pub fn new(text: String) -> Self {
        let cursor = std::io::Cursor::new(text);
        let reader = io::BufReader::new(cursor);
        StarTextReader {
            iter: Some(StarBufIter::new(reader)),
        }
    }

    pub fn next_block(&mut self) -> PyResult<Option<DataBlock>> {
        match self.iter.as_mut() {
            Some(it) => match it.next() {
                Some(Ok(block)) => Ok(Some(block)),
                Some(Err(e)) => Err(PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string())),
                None => Ok(None),
            }
            None => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Reader has been closed",
            )),
        }
    }

    pub fn close(&mut self) {
        // Explicitly drop the iterator to close the reader
        self.iter = None;
    }
}

/// An iterator over STAR data blocks that read entire block at a time
pub struct StarBufIter<R: io::BufRead> {
    reader: R,
    buf: String,
    line_remained: String,
}

impl<R: io::BufRead> StarBufIter<R> {
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            buf: String::new(),
            line_remained: String::new(),
        }
    }
}

impl<R: BufRead> Iterator for StarBufIter<R> {
    type Item = std::io::Result<DataBlock>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            self.buf.clear();

            let read_line_result = if self.line_remained.is_empty() {
                self.reader.read_line(&mut self.buf)
            } else {
                self.buf.push_str(&self.line_remained);
                self.line_remained.clear();
                Ok(self.buf.len())
            };

            match read_line_result {
                Ok(0) => {
                    return None;  // EOF
                }
                Ok(_) => {
                    let line = remove_comment(&self.buf).trim_end().to_string();
                    if line.is_empty() {
                        continue; // Skip empty lines
                    } else if line.starts_with("data_") {
                        // Start of a new data block
                        let data_block_name = line[5..].to_string();
                        let returned = parse_block(&mut self.reader, usize::MAX);
                        match returned {
                            Ok(returned) => {
                                let block = DataBlock::new(data_block_name.clone(), returned.block_data);
                                self.line_remained = returned.line_remained;
                                if block.block_type.is_eof() {
                                    return None;
                                }
                                return Some(Ok(block));
                            }
                            Err(e) => return Some(Err(e)),
                        }
                    } else {
                        continue;
                    }
                }
                Err(e) => return Some(Err(e)),
            };
        }
    }
}

fn parse_block<R: io::BufRead>(
    mut reader: &mut R,
    max_num_rows: usize,
) -> io::Result<ParsedBlock> {
    loop {
        let mut buf = String::new();
        match reader.read_line(&mut buf) {
            Ok(0) => return Ok(ParsedBlock::new("".to_string(), BlockData::EOF)),  // EOF
            Ok(_) => {
                let line = remove_comment(&buf).trim_end().to_string();
                if line.is_empty() {
                    continue; // Skip empty lines
                } else if line.starts_with("loop_") {
                    let (rem, columns) = parse_loop_block(&mut reader, max_num_rows)?;
                    return Ok(ParsedBlock::new(rem, BlockData::Loop(columns)))
                } else if line.starts_with("_") {
                    let scalar_first = Scalar::from_line(&line);
                    let (rem, scalars) = parse_simple_block(&mut reader)?;
                    let mut all_scalars = vec![scalar_first];
                    all_scalars.extend(scalars);
                    return Ok(ParsedBlock::new(rem, BlockData::Simple(all_scalars)))
                } else if line.starts_with("data_") {
                    // Start of next block
                    return Ok(ParsedBlock::new(buf.to_string(), BlockData::Simple(Vec::new())));
                }
                else {
                    // Unexpected line, stop parsing
                    return Err(err_internal());
                }
            }
            Err(_) => {
                return Err(err_unexpected_line(&buf));
            },
        }
    }
}


/// Parse a simple data block from the reader
///
/// A simple data block consists of lines starting with '_' as follows:
/// _column_1 value_1
/// _column_2 value_2
fn parse_simple_block<R: io::BufRead>(reader: &mut R) -> io::Result<(String, Vec<Scalar>)> {
    let mut scalars = Vec::new();
    let line_remained = loop {
        let mut buf = String::new();
        match reader.read_line(&mut buf) {
            Ok(0) => break "".to_string(), // EOF
            Ok(_) => {
                let buf_trim = buf.trim_end();
                if buf_trim.is_empty() {
                    break buf_trim.to_string(); // End of block
                } else if buf_trim.starts_with("_") {
                    let line = remove_comment(&buf_trim);
                    let scalar = Scalar::from_line(line);
                    scalars.push(scalar);
                } else if buf_trim.starts_with("#") {
                    continue; // Skip comments
                } else if buf_trim.starts_with("data_"){
                    break buf.to_string(); // Start of next block
                } else {
                    return Err(err_unexpected_line(&buf));
                }
            }
            Err(e) => return Err(e),
        }
    };
    Ok((line_remained, scalars))
}

fn parse_loop_block<R: io::BufRead>(
    reader: &mut R,
    max_num_rows: usize,
) -> io::Result<(String, LoopData)> {
    let mut column_names = Vec::<String>::new();

    // Parse column names
    let mut last_line = loop {
        let mut buf = String::new();
        match reader.read_line(&mut buf) {
            Ok(0) => {
                return Ok((
                    "".to_string(),
                     LoopData::new(column_names, "".to_string(), 0)
                )); // EOF
            }
            Ok(_) => {
                let buf_trim = buf.trim_end();
                if buf_trim.is_empty() {
                    continue; // Skip empty lines
                } else if buf_trim.starts_with("_") {
                    column_names.push(remove_comment(&buf_trim[1..]).to_string());
                } else {
                    // Reached next data section
                    break buf.to_string();
                }
            }
            Err(e) => return Err(e),
        }
    };

    if last_line.starts_with("data_") {
        // This happens when there is no data row in the loop block
        return Ok((
            last_line,
            LoopData::new(column_names, "".to_string(), 0),
        ));
    }

    // Parse data rows
    let mut nrows = 1;
    let mut buf = String::new();
    let line_remained = loop {
        // NOTE: Using push_str is more efficient than Vec<String>, but less flexible
        // if we want to access individual rows or to insert rows later.
        buf.clear();
        match reader.read_line(&mut buf) {
            Ok(0) => break "".to_string(), // EOF
            Ok(_) => {
                if buf.trim_end().is_empty() {
                    break "".to_string(); // End of block
                } else if buf.starts_with("data_") {
                    break buf.to_string(); // Start of next block
                } else {
                    last_line.push_str(&buf);
                    nrows += 1;
                }
            }
            Err(e) => return Err(e),
        }
        if nrows >= max_num_rows {
            break "".to_string(); // Reached max number of rows
        }
    };
    Ok((line_remained, LoopData::new(column_names, last_line, nrows)))
}

fn remove_comment(line: &str) -> &str {
    if let Some(pos) = line.find('#') {
        &line[..pos].trim_end()
    } else {
        line.trim_end()
    }
}

fn err_unexpected_line(buf: &str) -> io::Error {
    let msg = format!("Unexpected line while parsing block: {}", buf);
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

fn err_internal() -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        "Error reading line while parsing block",
    )
}

struct ParsedBlock {
    line_remained: String,
    block_data: BlockData,
}

impl ParsedBlock {
    fn new(line_remained: String, block_data: BlockData) -> Self {
        Self {
            line_remained,
            block_data,
        }
    }
}
