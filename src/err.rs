use std::io;

pub fn err_unexpected_line(buf: &str) -> io::Error {
    let msg = format!("Unexpected line while parsing block: {}", buf);
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

pub fn err_internal() -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        "Error reading line while parsing block",
    )
}

pub fn err_non_utf8(line: &str) -> io::Error {
    let msg = format!("Non-UTF8 line encountered: {}", line);
    io::Error::new(io::ErrorKind::InvalidData, msg)
}
