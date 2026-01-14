use pyo3::prelude::*;
use std::fs::File;
use std::io::{self, BufRead};
use std::usize;
use crate::blocks::{DataBlock, BlockData, Scalar, LoopData};
use crate::err::{err_internal, err_unexpected_line, err_non_utf8};

#[pyclass]
/// A reader for STAR files that iterate over data blocks
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
    // buf and line_remained will be used for reading lines.
    buf: Vec<u8>,
    line_remained: Vec<u8>,
}

const DEFAULT_CAPACITY: usize = 256;

impl<R: io::BufRead> StarBufIter<R> {
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            buf: Vec::with_capacity(DEFAULT_CAPACITY),
            line_remained: Vec::with_capacity(DEFAULT_CAPACITY),
        }
    }
}

impl<R: BufRead> Iterator for StarBufIter<R> {
    type Item = std::io::Result<DataBlock>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            self.buf.clear();

            let read_line_result = if self.line_remained.is_empty() {
                self.reader.read_until(b'\n', &mut self.buf)
            } else {
                self.buf.extend(&self.line_remained);
                self.line_remained.clear();
                Ok(self.buf.len())
            };

            match read_line_result {
                Ok(0) => {
                    return None;  // EOF
                }
                Ok(_) => {
                    let line_without_comment = remove_comment(&self.buf);
                    let line = line_without_comment.trim_ascii_end();
                    if line.is_empty() {
                        continue; // Skip empty lines
                    } else if line.starts_with(b"data_") {
                        // Start of a new data block
                        // SAFETY: data_ prefix is ASCII, so we can safely decode just the name part
                        let data_block_name = unsafe {
                            String::from_utf8_unchecked(line[5..].to_vec())
                        };
                        let returned = parse_block(&mut self.reader, usize::MAX);
                        match returned {
                            Ok(returned) => {
                                let block = DataBlock::new(data_block_name, returned.block_data);
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
        let mut buf = Vec::new();
        match reader.read_until(b'\n', &mut buf) {
            Ok(0) => return Ok(ParsedBlock::eof()),  // EOF
            Ok(_) => {
                let line = remove_comment(&buf).trim_ascii_end();
                if line.is_empty() {
                    continue; // Skip empty lines
                } else if line.starts_with(b"loop_") {
                    let (rem, loopdata) = parse_loop_block(&mut reader, max_num_rows)?;
                    return Ok(ParsedBlock::new(rem, BlockData::Loop(loopdata)))
                } else if line.starts_with(b"_") {
                    let line_str = std::str::from_utf8(&line)
                        .map_err(|_| err_non_utf8(&String::from_utf8_lossy(&line)))?;
                    let scalar_first = Scalar::from_line(line_str);
                    let (rem, scalars) = parse_simple_block(&mut reader)?;
                    let mut all_scalars = vec![scalar_first];
                    all_scalars.extend(scalars);
                    return Ok(ParsedBlock::new(rem, BlockData::Simple(all_scalars)))
                } else if line.starts_with(b"data_") {
                    // Start of next block
                    return Ok(ParsedBlock::new(buf, BlockData::Simple(Vec::new())));
                }
                else {
                    // Unexpected line, stop parsing
                    return Err(err_internal());
                }
            }
            Err(_) => {
                let line = String::from_utf8_lossy(&buf).to_string();
                return Err(err_unexpected_line(&line));
            },
        }
    }
}


/// Parse a simple data block from the reader
///
/// A simple data block consists of lines starting with '_' as follows:
/// _column_1 value_1
/// _column_2 value_2
fn parse_simple_block<R: io::BufRead>(reader: &mut R) -> io::Result<(Vec<u8>, Vec<Scalar>)> {
    let mut scalars = Vec::new();
    let line_remained = loop {
        let mut buf = Vec::new();
        match reader.read_until(b'\n', &mut buf) {
            Ok(0) => break Vec::new(), // EOF
            Ok(_) => {
                let buf_trim = buf.trim_ascii_end();
                if buf_trim.is_empty() {
                    break Vec::new(); // End of block
                } else if buf_trim.starts_with(b"_") {
                    let line = remove_comment(buf_trim.into());
                    let line_decoded = String::from_utf8_lossy(&line).to_string();
                    let scalar = Scalar::from_line(line_decoded.as_str());
                    scalars.push(scalar);
                } else if buf_trim.starts_with(b"#") {
                    continue; // Skip comments
                } else if buf_trim.starts_with(b"data_"){
                    break buf_trim.into(); // Start of next block
                } else {
                    let line = String::from_utf8_lossy(buf_trim).to_string();
                    return Err(err_unexpected_line(&line));
                }
            }
            Err(e) => return Err(e),
        }
    };
    Ok((line_remained, scalars))
}

/// Parse a loop data block from the reader
fn parse_loop_block<R: io::BufRead>(
    reader: &mut R,
    max_num_rows: usize,
) -> io::Result<(Vec<u8>, LoopData)> {
    let mut column_names = Vec::new();

    // Parse column names
    let mut buf = Vec::new();
    let mut last_line = loop {
        buf.clear();
        match reader.read_until(b'\n', &mut buf) {
            Ok(0) => {
                return Ok((
                    Vec::new(),
                    LoopData::new_empty(column_names)
                )); // EOF
            }
            Ok(_) => {
                let buf_trim = buf.trim_ascii_end();
                if buf_trim.is_empty() {
                    continue; // Skip empty lines
                } else if buf_trim.starts_with(b"_") {
                    let buf_vec = remove_comment(&buf_trim);
                    // SAFETY: Column names should be ASCII-compatible
                    let buf_str = unsafe { std::str::from_utf8_unchecked(&buf_vec) };
                    column_names.push(buf_str[1..].to_string());
                } else {
                    // Reached next data section
                    let mut rem = buf_trim.to_vec();
                    rem.push(b'\n');
                    break rem;
                }
            }
            Err(e) => return Err(e),
        }
    };

    if last_line.starts_with(b"data_") {
        // This happens when there is no data row in the loop block
        return Ok((
            last_line,
            LoopData::new_empty(column_names),
        ));
    }

    // Parse data rows
    let mut nrows = 1;
    let mut buf = Vec::with_capacity(last_line.len());
    let mut offsets = vec![0, last_line.len()];
    let line_remained = loop {
        buf.clear();
        match reader.read_until(b'\n', &mut buf) {
            Ok(0) => break Vec::new(), // EOF
            Ok(_) => {
                let buf_trim = buf.trim_ascii_end();
                if buf_trim.is_empty() {
                    break Vec::new(); // End of block
                } else if buf.starts_with(b"data_") {
                    break Vec::new(); // Start of next block
                } else {
                    last_line.extend(buf_trim);
                    last_line.push(b'\n');
                    offsets.push(last_line.len());
                    nrows += 1;
                }
            }
            Err(e) => return Err(e),
        }
        if nrows >= max_num_rows {
            break Vec::new(); // Reached max number of rows
        }
    };
    Ok((line_remained, LoopData::new(column_names, last_line, offsets)))
}

#[inline(always)]
fn remove_comment(line: &[u8]) -> &[u8] {
    if let Some(pos) = line.iter().position(|&c| c == b'#') {
        &line[..pos].trim_ascii_end()
    } else {
        line
    }
}

struct ParsedBlock {
    line_remained: Vec<u8>,
    block_data: BlockData,
}

impl ParsedBlock {
    fn new(line_remained: Vec<u8>, block_data: BlockData) -> Self {
        Self {
            line_remained,
            block_data,
        }
    }

    fn eof() -> Self {
        Self {
            line_remained: Vec::new(),
            block_data: BlockData::EOF,
        }
    }
}
