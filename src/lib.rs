use pyo3::prelude::*;
pub mod reader;
pub mod writer;
pub mod blocks;

/// A Python module implemented in Rust.
#[pymodule]
fn _starfile_rs_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<reader::StarReader>()?;
    m.add_class::<blocks::DataBlock>()?;
    m.add_class::<blocks::BlockType>()?;
    m.add_class::<writer::StarWriter>()?;
    Ok(())
}
