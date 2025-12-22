use pyo3::prelude::*;
use std::path::Path;
use std::fs::File;
use std::io;

#[pyclass]
pub struct StarWriter {
    writer: io::BufWriter<File>,
}


impl StarWriter {
    pub fn new(path: &Path) -> io::Result<Self> {
        let writer = File::create(path)?;
        Ok(StarWriter {writer: io::BufWriter::new(writer)})
    }
}
