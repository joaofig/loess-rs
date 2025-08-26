# loess-rs

A fast, lightweight implementation of the LOESS/LOWESS smoothing algorithm in Rust, powered by nalgebra. This repository provides a simple, dependency-light baseline you can use to smooth noisy 1D data with locally weighted regression.


## Highlights
- Pure Rust + nalgebra, no heavy ecosystem dependencies
- Tricubic kernel weights and k-nearest neighborhood window
- Linear and polynomial fit (configurable degree)
- Numerically stable via pseudo-inverse with configurable tolerance
- Small, readable codebase you can adapt to your needs


## What is LOESS/LOWESS?
LOESS (a.k.a. LOWESS) is a non-parametric smoothing method. For each x you want to predict, it fits a local polynomial on the k nearest neighbors, weighted by their distance to x (tricubic kernel here), and evaluates that polynomial at x. This yields a smooth curve that follows the trend of the data while being robust to noise and outliers.

- NIST overview: https://www.itl.nist.gov/div898/handbook/pmd/section1/dep/dep144.htm
- Wikipedia: https://en.wikipedia.org/wiki/Local_regression
- Kernel reference (tricubic): https://en.wikipedia.org/wiki/Kernel_(statistics)#Kernel_functions_in_common_use
- Author’s article: https://medium.com/@joao.figueira/loess-373d43b03564


## Getting started

### Prerequisites
- Rust toolchain (stable) with Cargo installed: https://www.rust-lang.org/tools/install

### Build & run
This repository includes a main.rs with a minimal example and a small benchmark-like loop.

- Build in debug:
  - `cargo build`
- Run example:
  - `cargo run`

You should see an elapsed time print indicating how long the example smoothing took.


## Library overview
The core logic lives in `src/main.rs` for simplicity. The main types and functions:

- Loess::new(xs, ys)
  - Normalizes X and Y to [0, 1] for numerical stability.
- Loess::estimate(x, window, use_matrix, degree) -> f64
  - x: query point (in original scale)
  - window: number of nearest neighbors to use (k)
  - use_matrix: when true (or degree > 1), uses matrix algebra fit; otherwise a fast weighted linear stats path
  - degree: local polynomial degree, e.g. 1 (linear), 2 (quadratic)

Supporting pieces:
- Tricubic kernel for weights
- Distance-based neighbor selection (k-nearest window around x)
- Weighted least squares via normal equations and pseudo-inverse (nalgebra)

Note: In its current form, this project exposes the Loess type inside `main.rs`. If you want to consume it as a library (crate), consider moving the type into `src/lib.rs`, exposing a public API, and keeping a small `src/bin` example.


## Usage example
Below is a simplified outline based on `main.rs`:

```rust
use nalgebra::DVector;

// Prepare your data (monotonic X is typical but not strictly required)
let xx = DVector::from_vec(vec![0.5, 2.0, 2.6, 3.4, 4.3]);
let yy = DVector::from_vec(vec![18.6, 103.5, 150.3, 190.5, 208.7]);

let loess = Loess::new(&xx, &yy);

let window = 7;       // k-nearest neighbors
let degree = 1;       // local linear
let use_matrix = true; // matrix fit path (recommended for degree >= 1)

let x_query = 3.0;
let y_smooth = loess.estimate(x_query, window, use_matrix, degree);
println!("smoothed y at {} => {}", x_query, y_smooth);
```


## Choosing parameters
- window (k): larger windows give smoother curves but may over-smooth details; smaller windows follow data more closely but can be noisier. A common heuristic is k ~ 10–30% of your dataset size.
- degree: 1 (linear) works well in many cases; 2 (quadratic) can follow curvature better but may be more sensitive to noise.
- use_matrix: kept for flexibility/performance. For degree > 1, matrix path is selected automatically; for degree == 1 you can choose the stats path (fast) or matrix path (general).


## Performance notes
- Normalization helps numerical stability.
- The implementation uses a pseudo-inverse with a small tolerance (1e-5). For very ill-conditioned local fits, tune that threshold or try a ridge term.
- For large datasets or many query points, consider:
  - Precomputing neighbor indices (e.g., using a sliding window for sorted X)
  - Parallelizing queries with rayon
  - Moving the algorithm into lib.rs and writing benchmarks in `benches/`


## Roadmap
- Extract Loess into a library crate (src/lib.rs) with a clean public API
- Add examples and docs.rs documentation
- Optional robustness iterations (reweighting by residuals)
- Optional span parameter as alternative to fixed window size
- Add tests and criterion benchmarks


## Related work
- Python version by the same author: https://github.com/joaofig/pyloess


## Contributing
Contributions are welcome! Feel free to open issues and PRs. If you plan a bigger change, please start a discussion first so we can align on the API and scope.


## License
This project is licensed under the MIT License. See the LICENSE file for details.
