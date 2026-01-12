use pyo3::prelude::*;
pub mod reader;
pub mod blocks;
pub mod err;

/// A Python module implemented in Rust.
#[pymodule]
fn _starfile_rs_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<reader::StarReader>()?;
    m.add_class::<reader::StarTextReader>()?;
    m.add_class::<blocks::DataBlock>()?;
    m.add_class::<blocks::BlockType>()?;

    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}
