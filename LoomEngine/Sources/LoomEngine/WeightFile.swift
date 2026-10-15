import Foundation

/// A tensor read from a `.loom` file: row-major Float storage plus its shape.
public struct Tensor: Sendable {
  public var shape: [Int]
  public var data: [Float]

  public init(shape: [Int], data: [Float]) {
    precondition(shape.reduce(1, *) == data.count, "shape \(shape) does not match \(data.count) values")
    self.shape = shape
    self.data = data
  }

  public init(zeros shape: [Int]) {
    self.shape = shape
    self.data = [Float](repeating: 0, count: shape.reduce(1, *))
  }

  public var count: Int { data.count }
  public var rows: Int { shape.count == 2 ? shape[0] : 1 }
  public var cols: Int { shape.last ?? 1 }

  public subscript(row: Int) -> ArraySlice<Float> {
    data[(row * cols)..<((row + 1) * cols)]
  }
}

/// The `.loom` container written by `Tools/train/loomio.py`:
/// 8-byte magic, a UInt32 header length, a JSON header, padding to 16 bytes, then raw tensors.
public struct WeightFile: Sendable {
  public let meta: [String: JSONValue]
  public let tensors: [String: Tensor]

  public enum Failure: Error, CustomStringConvertible {
    case badMagic, badHeader(String), missing(String)
    public var description: String {
      switch self {
      case .badMagic: "Not a .loom file"
      case .badHeader(let why): "Unreadable .loom header: \(why)"
      case .missing(let name): "The .loom file has no tensor named \(name)"
      }
    }
  }

  public init(url: URL) throws {
    try self.init(data: Data(contentsOf: url, options: .mappedIfSafe))
  }

  public init(data: Data) throws {
    guard data.count > 12, data.prefix(8) == Data("LOOMv1\0\0".utf8) else { throw Failure.badMagic }
    let headerLength = data.withUnsafeBytes { Int(UInt32(littleEndian: $0.loadUnaligned(fromByteOffset: 8, as: UInt32.self))) }
    guard 12 + headerLength <= data.count else { throw Failure.badHeader("length") }
    let headerData = data.subdata(in: 12..<(12 + headerLength))
    guard case .object(let header) = try JSONDecoder().decode(JSONValue.self, from: headerData),
      case .object(let meta) = header["meta"] ?? .object([:]),
      case .array(let table) = header["tensors"] ?? .array([])
    else { throw Failure.badHeader("structure") }
    var base = 12 + headerLength
    base += (16 - base % 16) % 16
    var tensors: [String: Tensor] = [:]
    try data.withUnsafeBytes { raw in
      for entry in table {
        guard case .object(let t) = entry, case .string(let name) = t["name"],
          case .array(let shapeValues) = t["shape"], let offset = t["offset"]?.number
        else { throw Failure.badHeader("tensor entry") }
        let shape = shapeValues.compactMap { $0.number.map(Int.init) }
        let count = shape.reduce(1, *)
        let start = base + Int(offset)
        let half = t["dtype"]?.string == "f16"
        var values = [Float](repeating: 0, count: count)
        if half {
          guard start + count * 2 <= raw.count else { throw Failure.badHeader("\(name) overruns the file") }
          for i in 0..<count {
            let bits = UInt16(littleEndian: raw.loadUnaligned(fromByteOffset: start + i * 2, as: UInt16.self))
            values[i] = Float(Float16(bitPattern: bits))
          }
        } else {
          guard start + count * 4 <= raw.count else { throw Failure.badHeader("\(name) overruns the file") }
          values.withUnsafeMutableBytes { dst in
            dst.copyMemory(from: UnsafeRawBufferPointer(rebasing: raw[start..<(start + count * 4)]))
          }
        }
        tensors[name] = Tensor(shape: shape, data: values)
      }
    }
    self.meta = meta
    self.tensors = tensors
  }

  public func tensor(_ name: String) throws -> Tensor {
    guard let t = tensors[name] else { throw Failure.missing(name) }
    return t
  }
}

/// Minimal JSON tree for headers and fixtures.
public enum JSONValue: Codable, Sendable, Equatable {
  case null, bool(Bool), number(Double), string(String), array([JSONValue]), object([String: JSONValue])

  public init(from decoder: Decoder) throws {
    let c = try decoder.singleValueContainer()
    if c.decodeNil() { self = .null }
    else if let b = try? c.decode(Bool.self) { self = .bool(b) }
    else if let n = try? c.decode(Double.self) { self = .number(n) }
    else if let s = try? c.decode(String.self) { self = .string(s) }
    else if let a = try? c.decode([JSONValue].self) { self = .array(a) }
    else { self = .object(try c.decode([String: JSONValue].self)) }
  }

  public func encode(to encoder: Encoder) throws {
    var c = encoder.singleValueContainer()
    switch self {
    case .null: try c.encodeNil()
    case .bool(let b): try c.encode(b)
    case .number(let n): try c.encode(n)
    case .string(let s): try c.encode(s)
    case .array(let a): try c.encode(a)
    case .object(let o): try c.encode(o)
    }
  }

  public var number: Double? { if case .number(let n) = self { n } else { nil } }
  public var string: String? { if case .string(let s) = self { s } else { nil } }
  public var array: [JSONValue]? { if case .array(let a) = self { a } else { nil } }
  public var object: [String: JSONValue]? { if case .object(let o) = self { o } else { nil } }
  public subscript(key: String) -> JSONValue? { object?[key] }
  public var doubles: [Double] { array?.compactMap(\.number) ?? [] }
  public var floats: [Float] { array?.compactMap { $0.number.map(Float.init) } ?? [] }
}
