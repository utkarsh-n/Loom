// Draws Loom's app icon: a causal attention matrix in neon on black — each row a token, each cell how
// much it attends to an earlier one, rows coloured by the pipeline's stages. 1024 × 1024, no alpha.
//   swift Tools/make-icon.swift Loom/Resources/Assets.xcassets/AppIcon.appiconset/icon-1024.png
import AppKit
import CoreGraphics

let size = 1024
let out = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "icon-1024.png"
let cs = CGColorSpace(name: CGColorSpace.sRGB)!
let ctx = CGContext(data: nil, width: size, height: size, bitsPerComponent: 8, bytesPerRow: 0, space: cs,
  bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
func c(_ r: CGFloat, _ g: CGFloat, _ b: CGFloat, _ a: CGFloat = 1) -> CGColor { CGColor(colorSpace: cs, components: [r, g, b, a])! }
let S = CGFloat(size)
ctx.setFillColor(c(0.008, 0.010, 0.018)); ctx.fill(CGRect(x: 0, y: 0, width: S, height: S))
let glow = CGGradient(colorsSpace: cs, colors: [c(0.10, 0.07, 0.20), c(0.008, 0.010, 0.018)] as CFArray, locations: [0, 1])!
ctx.drawRadialGradient(glow, startCenter: CGPoint(x: S / 2, y: S / 2), startRadius: 0, endCenter: CGPoint(x: S / 2, y: S / 2), endRadius: S * 0.6, options: [])

// Row colours: input cyan, tokenize green, embedding yellow, predict orange, sample pink.
let rows: [(CGFloat, CGFloat, CGFloat)] = [(0.30, 0.86, 1.0), (0.42, 1.0, 0.50), (1.0, 0.86, 0.28), (1.0, 0.58, 0.24), (1.0, 0.38, 0.74)]
// Attention weights: each row sums to 1 over the cells at or before the diagonal.
let w: [[CGFloat]] = [[1], [0.35, 0.65], [0.15, 0.55, 0.30], [0.10, 0.15, 0.25, 0.50], [0.30, 0.05, 0.10, 0.15, 0.40]]
let n = 5
let grid = S * 0.64
let cell = grid / CGFloat(n)
let origin = CGPoint(x: (S - grid) / 2, y: (S - grid) / 2)
for i in 0..<n {
  for j in 0..<n {
    // CoreGraphics' y runs up; row 0 is drawn at the top.
    let rect = CGRect(x: origin.x + CGFloat(j) * cell + cell * 0.09, y: origin.y + CGFloat(n - 1 - i) * cell + cell * 0.09,
      width: cell * 0.82, height: cell * 0.82)
    let path = CGPath(roundedRect: rect, cornerWidth: cell * 0.14, cornerHeight: cell * 0.14, transform: nil)
    let (r, g, b) = rows[i]
    if j <= i {
      let a = w[i][j]
      for k in stride(from: 4, through: 1, by: -1) {
        let grow = CGFloat(k) * cell * 0.06 * a
        let gp = CGPath(roundedRect: rect.insetBy(dx: -grow, dy: -grow), cornerWidth: cell * 0.2, cornerHeight: cell * 0.2, transform: nil)
        ctx.addPath(gp); ctx.setFillColor(c(r, g, b, 0.06 * a)); ctx.fillPath()
      }
      ctx.addPath(path); ctx.setFillColor(c(r, g, b, 0.12 + 0.88 * a)); ctx.fillPath()
      ctx.addPath(path); ctx.setStrokeColor(c(r, g, b, 0.9)); ctx.setLineWidth(5); ctx.strokePath()
    } else {
      ctx.addPath(path); ctx.setStrokeColor(c(1, 1, 1, 0.08)); ctx.setLineWidth(4); ctx.strokePath()
    }
  }
}
let image = ctx.makeImage()!
try! NSBitmapImageRep(cgImage: image).representation(using: .png, properties: [:])!.write(to: URL(fileURLWithPath: out))
print("wrote", out)
