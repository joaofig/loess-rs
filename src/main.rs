use std::ops::Mul;
use ndarray::{Array1, array};
use ndarray_stats::QuantileExt;


fn tricubic(x: &Array1<f64>) -> Array1<f64> {
    let dim = x.raw_dim();
    let mut y: Array1<f64> = Array1::<f64>::zeros(dim);

    for (i, v) in x.indexed_iter() {
        if *v >= -1.0 && *v <= 1.0 {
            y[i] = (1.0 - v.abs().powi(3)).powi(3)
        }
    }
    y
}


fn normalize(x: &Array1<f64>) -> (Array1<f64>, f64, f64) {
    let dim = x.raw_dim();
    let mut y: Array1<f64> = Array1::<f64>::zeros(dim);
    let mut min_val: f64 = x[0];
    let mut max_val: f64 = x[0];

    for &v in x.into_iter() {
        if v > max_val { max_val = v}
        if v < min_val { min_val = v}
    }
    for (i, &value) in x.indexed_iter() {
        y[i] = (value - min_val) / (max_val - min_val);
    }
    (y, min_val, max_val)
}


fn get_min_range(distances: &Array1<f64>, window: usize) -> Array1<usize> {
    let min_idx: usize = distances.argmin().unwrap();
    let n: usize = distances.len();
    let range: Array1<usize>;

    if min_idx == 0 {
        range = Array1::<usize>::from_iter(0..window);
    } else if min_idx == n - 1 {
        range = Array1::<usize>::from_iter(n - window..window);
    } else {
        let mut min_range: Vec<usize> = vec![min_idx];
        let mut l: usize = 1;
        while l < window {
            let i0: usize = min_range[0];
            let i1: usize = min_range[l-1];

            if i0 == 0 {
                min_range.push(i1 + 1);
            } else if (i1 == n - 1) || (distances[i0 - 1] < distances[i1 + 1]) {
                min_range.insert(0, i0 - 1);
            } else {
                min_range.push(i1 + 1);
            }
            l += 1;
        }
        range = Array1::<usize>::from_vec(min_range);
    }
    range
}


fn select_indices(values: &Array1<f64>, indices: &Array1<usize>) -> Array1<f64> {
    let filtered: Vec<f64> = Vec::<f64>::from_iter(indices.iter().map(|&ix| values[ix]));
    Array1::<f64>::from_vec(filtered)
}


fn get_weights(distances: &Array1<f64>, min_range: &Array1<usize>) -> Array1<f64> {
    let selection: Array1<f64> = select_indices(distances, min_range);
    let max_distance: f64 = *distances.max().unwrap();
    let norm_distances: Array1<f64> = selection / max_distance;
    tricubic(&norm_distances)
}


pub struct Loess {
    xx: Array1<f64>,
    yy: Array1<f64>,
    deg: i32,
    min_y: f64,
    max_y: f64,
    min_x: f64,
    max_x: f64,
}


impl Loess {
    fn new(xs: &Array1<f64>, ys: &Array1<f64>, degree: i32) -> Self {
        let (norm_xs, min_x, max_x) = normalize(xs);
        let (norm_ys, min_y, max_y) = normalize(ys);
        Loess {
            xx: norm_xs,
            yy: norm_ys,
            deg: degree,
            min_y: min_y,
            max_y: max_y,
            min_x: min_x,
            max_x: max_x,
        }
    }

    fn normalize_x(&self, x: f64) -> f64 {
        (x - self.min_x) / (self.max_x - self.min_x)
    }

    fn denormalize_y(&self, y: f64) -> f64 {
        y * (self.max_y - self.min_y) + self.min_y
    }

    fn estimate(&self, x: f64,
                window: usize,
                use_matrix: bool,
                degree: i32
                ) -> f64 {
        let mut y: f64 = 0.0;
        let n_x: f64 = self.normalize_x(x);
        let distances: Array1<f64> = (&self.xx - n_x).mapv(|v| v.abs());
        let min_range: Array1<usize> = get_min_range(&distances, window);
        let weights: Array1<f64> = get_weights(&distances, &min_range);

        if use_matrix || degree > 1 {

        } else {
            let xx = select_indices(&self.xx, &min_range);
            let yy = select_indices(&self.yy, &min_range);
            let sum_weight = weights.sum();
            let sum_weight_x = xx.dot(&weights);
            let sum_weight_y = yy.dot(&weights);
            let sum_weight_x2 = xx.clone().mul(&xx).dot(&weights);
            let sum_weight_xy = xx.clone().mul(&yy).dot(&weights);

            let mean_x = sum_weight_x / sum_weight;
            let mean_y = sum_weight_y / sum_weight;

            let b = (sum_weight_xy - mean_x * mean_y * sum_weight) / 
                    (sum_weight_x2 - mean_x * mean_x * sum_weight);
            let a = mean_y - b * mean_x;
            y = a + b * n_x
        }

        self.denormalize_y(y)
    }
}


fn main() {
    let xx: Array1<f64> =
        array![0.5578196, 2.0217271, 2.5773252, 3.4140288, 4.3014084,
               4.7448394, 5.1073781, 6.5411662, 6.7216176, 7.2600583,
               8.1335874, 9.1224379, 11.9296663, 12.3797674, 13.2728619,
               14.2767453, 15.3731026, 15.6476637, 18.5605355, 18.5866354,
               18.7572812];
    let yy: Array1<f64> =
        array![18.63654, 103.49646, 150.35391, 190.51031, 208.70115,
               213.71135, 228.49353, 233.55387, 234.55054, 223.89225,
               227.68339, 223.91982, 168.01999, 164.95750, 152.61107,
               160.78742, 168.55567, 152.42658, 221.70702, 222.69040,
               243.18828];

    let loess: Loess = Loess::new(&xx, &yy, 1);

    for (i, &value) in xx.indexed_iter() {
        println!("xx[{}] = {}", i, value);
    }
}
